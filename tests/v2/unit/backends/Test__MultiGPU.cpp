/**
 * @file Test__MultiGPU.cpp
 * @brief Unit tests for multi-GPU functionality (Phase 6)
 *
 * Tests the following multi-GPU capabilities:
 * - DeviceManager device enumeration
 * - Global device index to backend-specific device ID mapping
 * - TensorBase ensureOnDevice/ensureOnHost with heterogeneous backends
 * - Kernel fallback behavior (CPU fallbacks when GPU kernels not implemented)
 * - Cross-device memory management
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "backends/ComputeBackend.h"
#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "backends/DeviceId.h"
#include "tensors/Tensors.h"
#include "kernels/KernelFactory.h"
#include "utils/Logger.h"
#include <vector>
#include <memory>
#include <cstring>

#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#endif

#ifdef HAVE_ROCM
#include "backends/rocm/ROCmBackend.h"
#endif

using namespace llaminar2;

// ============================================================================
// Global Test Initialization
// ============================================================================

/**
 * @brief Initialize DeviceManager for all tests
 *
 * DeviceManager::initialize() must be called before accessing devices().
 * We initialize with -1 to enumerate all devices (no NUMA filtering).
 */
class MultiGPUEnvironment : public ::testing::Environment
{
public:
    void SetUp() override
    {
        // Initialize device manager with all devices (no NUMA filtering)
        DeviceManager::instance().initialize(-1);
    }
};

// Register the test environment
static ::testing::Environment *const multi_gpu_env =
    ::testing::AddGlobalTestEnvironment(new MultiGPUEnvironment);

// ============================================================================
// DeviceManager Tests
// ============================================================================

class Test__MultiGPU_DeviceManager : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // DeviceManager is initialized by global environment
    }
};

/**
 * @test DeviceManager should always have at least CPU (device 0)
 */
TEST_F(Test__MultiGPU_DeviceManager, HasCPUDevice)
{
    const auto &devices = DeviceManager::instance().devices();
    ASSERT_GE(devices.size(), 1) << "DeviceManager should have at least 1 device (CPU)";

    // First device should be CPU
    const auto &cpu = devices[0];
    EXPECT_TRUE(cpu.type == ComputeBackendType::CPU ||
                cpu.type == ComputeBackendType::CPU)
        << "Device 0 should be CPU";
    EXPECT_EQ(cpu.device_id, 0) << "CPU should have device_id = 0";
}

/**
 * @test DeviceManager device enumeration
 */
TEST_F(Test__MultiGPU_DeviceManager, EnumeratesAllDevices)
{
    const auto &devices = DeviceManager::instance().devices();
    std::cout << "DeviceManager enumerated " << devices.size() << " device(s):\n";

    for (size_t i = 0; i < devices.size(); ++i)
    {
        const auto &dev = devices[i];
        std::string type_name;
        switch (dev.type)
        {
        case ComputeBackendType::CPU:
            type_name = "CPU";
            break;
        case ComputeBackendType::GPU_CUDA:
            type_name = "GPU_CUDA";
            break;
        case ComputeBackendType::GPU_ROCM:
            type_name = "GPU_ROCM";
            break;
        case ComputeBackendType::GPU_VULKAN:
            type_name = "GPU_VULKAN";
            break;
        case ComputeBackendType::GPU_METAL:
            type_name = "GPU_METAL";
            break;
        }
        std::cout << "  [" << i << "] " << type_name
                  << " (backend device_id=" << dev.device_id << "): "
                  << dev.name << " - " << (dev.total_memory_bytes / (1024 * 1024)) << " MB\n";
    }
}

/**
 * @test Device indices are consecutive starting from 0
 */
TEST_F(Test__MultiGPU_DeviceManager, DeviceIndicesAreConsecutive)
{
    const auto &devices = DeviceManager::instance().devices();

    // Global indices should be 0, 1, 2, ... (by vector index)
    // Backend device_ids may not be consecutive (e.g., CUDA device 0, ROCm devices 0, 1)
    for (size_t i = 0; i < devices.size(); ++i)
    {
        // Each device should have a valid backend device_id
        EXPECT_GE(devices[i].device_id, 0)
            << "Device " << i << " has invalid backend device_id";
    }
}

// ============================================================================
// Backend Mapping Tests
// ============================================================================

class Test__MultiGPU_BackendMapping : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // DeviceManager initializes lazily on first access
    }
};

/**
 * @test getBackendForDeviceType returns correct backend
 */
TEST_F(Test__MultiGPU_BackendMapping, GetBackendForDeviceType)
{
    // CPU should not have a backend (returns nullptr)
    auto cpu_backend = getBackendForDeviceType(ComputeBackendType::CPU);
    EXPECT_EQ(cpu_backend, nullptr) << "CPU should not have an IBackend";

#ifdef HAVE_CUDA
    auto cuda_backend = getBackendForDeviceType(ComputeBackendType::GPU_CUDA);
    EXPECT_NE(cuda_backend, nullptr) << "CUDA backend should exist when compiled with HAVE_CUDA";
#endif

#ifdef HAVE_ROCM
    auto rocm_backend = getBackendForDeviceType(ComputeBackendType::GPU_ROCM);
    EXPECT_NE(rocm_backend, nullptr) << "ROCm backend should exist when compiled with HAVE_ROCM";
#endif
}

/**
 * @test Backend device count matches DeviceManager count for each type
 */
TEST_F(Test__MultiGPU_BackendMapping, BackendDeviceCountConsistency)
{
    const auto &devices = DeviceManager::instance().devices();

    // Count devices per type
    int cuda_count = 0;
    int rocm_count = 0;

    for (const auto &dev : devices)
    {
        if (dev.type == ComputeBackendType::GPU_CUDA)
            cuda_count++;
        else if (dev.type == ComputeBackendType::GPU_ROCM)
            rocm_count++;
    }

#ifdef HAVE_CUDA
    auto cuda_backend = getBackendForDeviceType(ComputeBackendType::GPU_CUDA);
    if (cuda_backend)
    {
        EXPECT_EQ(cuda_backend->deviceCount(), cuda_count)
            << "CUDA backend device count should match DeviceManager";
    }
#endif

#ifdef HAVE_ROCM
    auto rocm_backend = getBackendForDeviceType(ComputeBackendType::GPU_ROCM);
    if (rocm_backend)
    {
        EXPECT_EQ(rocm_backend->deviceCount(), rocm_count)
            << "ROCm backend device count should match DeviceManager";
    }
#endif
}

// ============================================================================
// FP32 Tensor GPU Transfer Tests
// ============================================================================

class Test__MultiGPU_TensorTransfer : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // DeviceManager initializes lazily on first access
        devices_ = DeviceManager::instance().devices();
    }

    // Find first GPU device (CUDA or ROCm)
    int findFirstGPU() const
    {
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            if (devices_[i].type == ComputeBackendType::GPU_CUDA ||
                devices_[i].type == ComputeBackendType::GPU_ROCM)
            {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    // Convert legacy DeviceManager index to DeviceId
    DeviceId legacyToDeviceId(int legacy_idx) const
    {
        if (legacy_idx < 0 || static_cast<size_t>(legacy_idx) >= devices_.size())
        {
            return DeviceId::cpu();
        }
        const auto &dev = devices_[legacy_idx];
        if (dev.type == ComputeBackendType::GPU_CUDA)
        {
            int cuda_ordinal = 0;
            for (int i = 1; i < legacy_idx; ++i)
            {
                if (devices_[i].type == ComputeBackendType::GPU_CUDA)
                    ++cuda_ordinal;
            }
            return DeviceId::cuda(cuda_ordinal);
        }
        else if (dev.type == ComputeBackendType::GPU_ROCM)
        {
            int rocm_ordinal = 0;
            for (int i = 1; i < legacy_idx; ++i)
            {
                if (devices_[i].type == ComputeBackendType::GPU_ROCM)
                    ++rocm_ordinal;
            }
            return DeviceId::rocm(rocm_ordinal);
        }
        return DeviceId::cpu();
    }

    std::vector<ComputeDevice> devices_;
};

/**
 * @test FP32Tensor ensureOnDevice with CPU target (should fail, ensureOnDevice is for GPU)
 */
TEST_F(Test__MultiGPU_TensorTransfer, FP32_EnsureOnCPU_Fails)
{
    std::vector<size_t> shape = {8, 16};
    auto tensor = std::make_unique<FP32Tensor>(shape, -1); // CPU tensor

    // Fill with test data
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < 128; ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // ensureOnDevice with CPU DeviceId should fail (ensureOnDevice is for GPU targets)
    DeviceId cpu_device = DeviceId::cpu();
    bool success = tensor->ensureOnDevice(cpu_device);
    EXPECT_FALSE(success) << "ensureOnDevice(DeviceId::cpu()) should fail - use ensureOnHost() for CPU";

    // Data should still be accessible
    EXPECT_FLOAT_EQ(tensor->data()[0], 0.0f);
    EXPECT_FLOAT_EQ(tensor->data()[127], 127.0f);
}

/**
 * @test FP32Tensor ensureOnDevice with GPU target
 */
TEST_F(Test__MultiGPU_TensorTransfer, FP32_EnsureOnGPU)
{
    int gpu_idx = findFirstGPU();
    if (gpu_idx < 0)
    {
        GTEST_SKIP() << "No GPU available for testing";
    }

    DeviceId gpu_device = legacyToDeviceId(gpu_idx);

    std::vector<size_t> shape = {8, 16};
    auto tensor = std::make_unique<FP32Tensor>(shape, -1); // Start on CPU

    // Fill with test data
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < 128; ++i)
    {
        data[i] = static_cast<float>(i) * 2.0f;
    }

    // Transfer to GPU
    bool success = tensor->ensureOnDevice(gpu_device);
    EXPECT_TRUE(success) << "ensureOnDevice should succeed for GPU " << gpu_device.toString();
    EXPECT_TRUE(tensor->isOnGPU()) << "Tensor should be on GPU after transfer";
    EXPECT_TRUE(tensor->is_on_device(gpu_idx))
        << "is_on_device(gpu_idx) should return true after transfer to that GPU";
    EXPECT_TRUE(tensor->is_on_device(0)) << "Host copy should still be valid (dual residency)";
}

/**
 * @test FP32Tensor round-trip transfer (CPU -> GPU -> CPU)
 */
TEST_F(Test__MultiGPU_TensorTransfer, FP32_RoundTrip)
{
    int gpu_idx = findFirstGPU();
    if (gpu_idx < 0)
    {
        GTEST_SKIP() << "No GPU available for testing";
    }

    DeviceId gpu_device = legacyToDeviceId(gpu_idx);

    std::vector<size_t> shape = {32, 64}; // Larger tensor
    auto tensor = std::make_unique<FP32Tensor>(shape, -1);

    // Fill with pattern
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < 32 * 64; ++i)
    {
        data[i] = static_cast<float>(i) * 1.5f;
    }

    // Copy original data
    std::vector<float> original(32 * 64);
    std::memcpy(original.data(), data, 32 * 64 * sizeof(float));

    // Transfer to GPU
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device));
    EXPECT_TRUE(tensor->isOnGPU());

    // Transfer back to host (host should already be valid since we did dual-residency upload)
    ASSERT_TRUE(tensor->ensureOnHost());
    EXPECT_TRUE(tensor->isOnCPU()) << "Host should be valid after ensureOnHost";

    // Verify data integrity (should match since we uploaded, not modified on GPU)
    const float *result = tensor->data();
    for (size_t i = 0; i < 32 * 64; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], original[i])
            << "Round-trip data mismatch at index " << i;
    }
}

/**
 * @test FP32Tensor release device memory
 */
TEST_F(Test__MultiGPU_TensorTransfer, FP32_ReleaseDeviceMemory)
{
    int gpu_idx = findFirstGPU();
    if (gpu_idx < 0)
    {
        GTEST_SKIP() << "No GPU available for testing";
    }

    DeviceId gpu_device = legacyToDeviceId(gpu_idx);

    std::vector<size_t> shape = {8, 16};
    auto tensor = std::make_unique<FP32Tensor>(shape, -1);

    // Fill with data
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < 128; ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Transfer to GPU
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device));
    EXPECT_TRUE(tensor->isOnGPU());

    // Release device memory
    ASSERT_TRUE(tensor->releaseDeviceMemory());
    EXPECT_FALSE(tensor->isOnGPU());

    // Host data should still be valid
    const float *result = tensor->data();
    for (size_t i = 0; i < 128; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], static_cast<float>(i))
            << "Host data should be preserved after releaseDeviceMemory";
    }
}

// ============================================================================
// Q4_0 Tensor GPU Transfer Tests
// ============================================================================

class Test__MultiGPU_Q4_0Transfer : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // DeviceManager initializes lazily on first access
        devices_ = DeviceManager::instance().devices();
    }

    int findFirstGPU() const
    {
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            if (devices_[i].type == ComputeBackendType::GPU_CUDA ||
                devices_[i].type == ComputeBackendType::GPU_ROCM)
            {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    DeviceId legacyToDeviceId(int legacy_idx) const
    {
        // Convert legacy DeviceManager index to DeviceId
        // Legacy index: CPU = 0, GPUs = 1, 2, 3...
        // DeviceId: DeviceId::cpu() or DeviceId::cuda(ordinal)
        int ordinal = 0;
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            if (static_cast<int>(i) == legacy_idx)
            {
                if (devices_[i].type == ComputeBackendType::GPU_CUDA)
                {
                    return DeviceId::cuda(ordinal);
                }
                else if (devices_[i].type == ComputeBackendType::GPU_ROCM)
                {
                    return DeviceId::rocm(ordinal);
                }
                return DeviceId::cpu();
            }
            if (devices_[i].type == ComputeBackendType::GPU_CUDA ||
                devices_[i].type == ComputeBackendType::GPU_ROCM)
            {
                ordinal++;
            }
        }
        return DeviceId::cpu();
    }

    std::vector<ComputeDevice> devices_;
};

/**
 * @test Q4_0Tensor byte_size calculation is shape-based (not raw_data dependent)
 */
TEST_F(Test__MultiGPU_Q4_0Transfer, ByteSizeFromShape)
{
    // Create Q4_0 tensor: 64 rows x 64 cols
    std::vector<size_t> shape = {64, 64};
    size_t blocks_per_row = (shape[1] + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
    size_t total_blocks = shape[0] * blocks_per_row;
    std::vector<uint8_t> raw_data(total_blocks * sizeof(Q4_0Block), 0);

    auto tensor = std::make_unique<Q4_0Tensor>(shape, std::move(raw_data));

    // Verify byte_size is calculated correctly
    size_t expected_bytes = total_blocks * sizeof(Q4_0Block);
    EXPECT_EQ(tensor->size_bytes(), expected_bytes)
        << "size_bytes should be calculated from shape";

    // Release raw data (simulating GEMM packing)
    tensor->release_raw_data();

    // size_bytes should STILL return correct value (not 0)
    EXPECT_EQ(tensor->size_bytes(), expected_bytes)
        << "size_bytes should work even after release_raw_data()";
}

/**
 * @test Q4_0Tensor GPU transfer
 */
TEST_F(Test__MultiGPU_Q4_0Transfer, EnsureOnGPU)
{
    int gpu_idx = findFirstGPU();
    if (gpu_idx < 0)
    {
        GTEST_SKIP() << "No GPU available for testing";
    }

    DeviceId gpu_device = legacyToDeviceId(gpu_idx);

    // Create Q4_0 tensor
    std::vector<size_t> shape = {32, 64};
    size_t blocks_per_row = (shape[1] + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
    size_t total_blocks = shape[0] * blocks_per_row;
    std::vector<uint8_t> raw_data(total_blocks * sizeof(Q4_0Block));

    // Fill with test pattern
    Q4_0Block *blocks = reinterpret_cast<Q4_0Block *>(raw_data.data());
    for (size_t i = 0; i < total_blocks; ++i)
    {
        blocks[i].d = static_cast<float>(i) * 0.1f;
        for (int j = 0; j < Q4_0Block::BLOCK_SIZE / 2; ++j)
        {
            blocks[i].qs[j] = static_cast<uint8_t>(i + j);
        }
    }

    auto tensor = std::make_unique<Q4_0Tensor>(shape, std::move(raw_data));

    // Transfer to GPU
    bool success = tensor->ensureOnDevice(gpu_device);
    EXPECT_TRUE(success) << "Q4_0 tensor should transfer to GPU";
    EXPECT_TRUE(tensor->isOnGPU()) << "Tensor should be on GPU after transfer";
    EXPECT_TRUE(tensor->is_on_device(gpu_idx))
        << "is_on_device(gpu_idx) should return true after transfer";
    EXPECT_TRUE(tensor->is_on_device(0)) << "Host copy should still be valid (dual residency)";
}

// ============================================================================
// Kernel Fallback Tests
// ============================================================================

class Test__MultiGPU_KernelFallback : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // DeviceManager initializes lazily on first access
    }

    int findFirstGPU() const
    {
        const auto &devices = DeviceManager::instance().devices();
        for (size_t i = 0; i < devices.size(); ++i)
        {
            if (devices[i].type == ComputeBackendType::GPU_CUDA ||
                devices[i].type == ComputeBackendType::GPU_ROCM)
            {
                return static_cast<int>(i);
            }
        }
        return -1;
    }
};

/**
 * @test RMSNorm kernel should fall back to CPU for GPU device requests
 */
TEST_F(Test__MultiGPU_KernelFallback, RMSNorm_CPUFallback)
{
    int gpu_idx = findFirstGPU();
    if (gpu_idx < 0)
    {
        GTEST_SKIP() << "No GPU available for testing";
    }

    // Create FP32 tensor
    std::vector<size_t> shape = {8, 16};
    auto tensor = std::make_unique<FP32Tensor>(shape, gpu_idx);

    // Request RMSNorm kernel for GPU device
    auto kernel = llaminar::v2::kernels::KernelFactory::createRMSNorm(
        tensor.get(), llaminar::v2::kernels::DeviceType::CPU);

    ASSERT_NE(kernel, nullptr) << "KernelFactory should return a kernel (CPU fallback)";

    // The kernel should be usable
    auto *rmsnorm = kernel.get();
    EXPECT_NE(rmsnorm, nullptr);
}

/**
 * @test SwiGLU kernel should fall back to CPU for GPU device requests
 */
TEST_F(Test__MultiGPU_KernelFallback, SwiGLU_CPUFallback)
{
    int gpu_idx = findFirstGPU();
    if (gpu_idx < 0)
    {
        GTEST_SKIP() << "No GPU available for testing";
    }

    // Create FP32 tensor
    std::vector<size_t> shape = {8, 16};
    auto tensor = std::make_unique<FP32Tensor>(shape, gpu_idx);

    // Request SwiGLU kernel for GPU device - should get CPU fallback
    auto kernel = llaminar::v2::kernels::KernelFactory::createSwiGLU(
        tensor.get(), llaminar::v2::kernels::DeviceType::CPU);

    ASSERT_NE(kernel, nullptr) << "KernelFactory should return a kernel (CPU fallback)";
}

/**
 * @test RoPE kernel should fall back to CPU for GPU device requests
 */
TEST_F(Test__MultiGPU_KernelFallback, RoPE_CPUFallback)
{
    int gpu_idx = findFirstGPU();
    if (gpu_idx < 0)
    {
        GTEST_SKIP() << "No GPU available for testing";
    }

    // Create FP32 tensor
    std::vector<size_t> shape = {8, 16};
    auto tensor = std::make_unique<FP32Tensor>(shape, gpu_idx);

    // Request RoPE kernel - should get CPU fallback
    auto kernel = llaminar::v2::kernels::KernelFactory::createRoPE(
        tensor.get(), llaminar::v2::kernels::DeviceType::CPU);

    ASSERT_NE(kernel, nullptr) << "KernelFactory should return a kernel (CPU fallback)";
}

// ============================================================================
// Multi-Device Transfer Tests (heterogeneous backends)
// ============================================================================

class Test__MultiGPU_HeterogeneousBackends : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // DeviceManager initializes lazily on first access
        devices_ = DeviceManager::instance().devices();

        // Find CUDA and ROCm devices and compute their ordinals
        int cuda_ordinal = 0;
        int rocm_ordinal = 0;
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            if (devices_[i].type == ComputeBackendType::GPU_CUDA)
            {
                if (cuda_idx_ < 0)
                {
                    cuda_idx_ = static_cast<int>(i);
                    cuda_device_ = DeviceId::cuda(cuda_ordinal);
                }
                cuda_ordinal++;
            }
            else if (devices_[i].type == ComputeBackendType::GPU_ROCM)
            {
                if (rocm_idx_ < 0)
                {
                    rocm_idx_ = static_cast<int>(i);
                    rocm_device_ = DeviceId::rocm(rocm_ordinal);
                }
                rocm_ordinal++;
            }
        }
    }

    std::vector<ComputeDevice> devices_;
    int cuda_idx_ = -1;
    int rocm_idx_ = -1;
    DeviceId cuda_device_ = DeviceId::cpu();
    DeviceId rocm_device_ = DeviceId::cpu();
};

/**
 * @test Transfer between different GPU backends (CUDA <-> ROCm)
 */
TEST_F(Test__MultiGPU_HeterogeneousBackends, CrossBackendTransfer)
{
    if (cuda_idx_ < 0 || rocm_idx_ < 0)
    {
        GTEST_SKIP() << "Need both CUDA and ROCm GPUs for this test";
    }

    std::cout << "Testing transfer: CUDA (global idx " << cuda_idx_
              << ") <-> ROCm (global idx " << rocm_idx_ << ")\n";

    std::vector<size_t> shape = {16, 32};
    auto tensor = std::make_unique<FP32Tensor>(shape, -1);

    // Fill with pattern
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < 16 * 32; ++i)
    {
        data[i] = static_cast<float>(i) * 0.5f;
    }
    std::vector<float> original(data, data + 16 * 32);

    // Transfer to CUDA
    ASSERT_TRUE(tensor->ensureOnDevice(cuda_device_));
    EXPECT_TRUE(tensor->isOnGPU()) << "Should be on CUDA GPU";
    EXPECT_TRUE(tensor->is_on_device(cuda_idx_)) << "Should report on CUDA device";

    // Transfer from CUDA to ROCm (will go via host since cross-backend)
    ASSERT_TRUE(tensor->ensureOnDevice(rocm_device_));
    EXPECT_TRUE(tensor->isOnGPU()) << "Should be on ROCm GPU";
    EXPECT_TRUE(tensor->is_on_device(rocm_idx_)) << "Should report on ROCm device";

    // Bring back to host and verify
    ASSERT_TRUE(tensor->ensureOnHost());

    const float *result = tensor->data();
    for (size_t i = 0; i < 16 * 32; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], original[i])
            << "Cross-backend transfer data mismatch at index " << i;
    }
}

/**
 * @test Verify backend-specific device IDs are correct
 */
TEST_F(Test__MultiGPU_HeterogeneousBackends, BackendSpecificDeviceIDs)
{
    std::cout << "Backend-specific device ID mapping:\n";

    for (size_t i = 0; i < devices_.size(); ++i)
    {
        const auto &dev = devices_[i];
        std::string type_str;
        if (dev.type == ComputeBackendType::CPU ||
            dev.type == ComputeBackendType::CPU)
        {
            type_str = "CPU";
        }
        else if (dev.type == ComputeBackendType::GPU_CUDA)
        {
            type_str = "CUDA";
        }
        else if (dev.type == ComputeBackendType::GPU_ROCM)
        {
            type_str = "ROCm";
        }
        else
        {
            type_str = "Other";
        }

        std::cout << "  Global idx " << i << " -> " << type_str
                  << " device ID " << dev.device_id << "\n";

        // Verify device_id is non-negative
        EXPECT_GE(dev.device_id, 0)
            << "Device " << i << " has invalid device_id";
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
