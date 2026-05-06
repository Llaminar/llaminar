/**
 * @file Test__KernelFactory.cpp
 * @brief Unit tests for KernelFactory centralized kernel dispatch
 * @author David Sanftenberg
 *
 * Tests cover:
 * 1. DeviceType enum and to_string conversion
 * 2. getDeviceType() device resolution
 * 3. createGemm() for all tensor types (CPU path)
 * 4. Error handling for invalid device indices
 * 5. Error handling for unsupported GPU backends
 */

#include <gtest/gtest.h>
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"
#include "tensors/TensorSlice.h"
#include "backends/ComputeBackend.h"

using namespace llaminar::v2::kernels;
using namespace llaminar2;

// ============================================================================
// DeviceType Enum Tests
// ============================================================================

class Test__KernelFactory : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Ensure DeviceManager is initialized (discovers CPU device)
        DeviceManager::instance();
    }
};

TEST_F(Test__KernelFactory, DeviceType_ToString_CPU)
{
    EXPECT_EQ(to_string(DeviceType::CPU), "CPU");
}

TEST_F(Test__KernelFactory, DeviceType_ToString_CUDA)
{
    EXPECT_EQ(to_string(DeviceType::CUDA), "CUDA");
}

TEST_F(Test__KernelFactory, DeviceType_ToString_ROCm)
{
    EXPECT_EQ(to_string(DeviceType::ROCm), "ROCm");
}

TEST_F(Test__KernelFactory, DeviceType_ToString_Vulkan)
{
    EXPECT_EQ(to_string(DeviceType::Vulkan), "Vulkan");
}

TEST_F(Test__KernelFactory, DeviceType_ToString_Metal)
{
    EXPECT_EQ(to_string(DeviceType::Metal), "Metal");
}

// ============================================================================
// getDeviceType() Tests
// ============================================================================

TEST_F(Test__KernelFactory, GetDeviceType_NegativeIndex_ReturnsCPU)
{
    // device_idx = -1 is the sentinel for "unspecified, use CPU"
    EXPECT_EQ(KernelFactory::getDeviceType(-1), DeviceType::CPU);
    EXPECT_EQ(KernelFactory::getDeviceType(-100), DeviceType::CPU);
}

TEST_F(Test__KernelFactory, GetDeviceType_ZeroIndex_ReturnsCPU)
{
    // In our test environment, device 0 is always CPU (OneDNN or MKL)
    auto &dm = DeviceManager::instance();
    if (dm.devices().empty())
    {
        GTEST_SKIP() << "DeviceManager has no devices (not initialized via MPI context)";
    }

    // Device 0 should be a CPU device
    DeviceType dev_type = KernelFactory::getDeviceType(0);
    EXPECT_EQ(dev_type, DeviceType::CPU);
}

TEST_F(Test__KernelFactory, GetDeviceType_InvalidIndex_Throws)
{
    auto &dm = DeviceManager::instance();
    size_t num_devices = dm.devices().size();

    // Index beyond available devices should throw
    EXPECT_THROW(
        KernelFactory::getDeviceType(static_cast<int>(num_devices + 100)),
        std::runtime_error);
}

// ============================================================================
// createGemm() Tests - Quantized Tensor Types (CPU)
// ============================================================================

// Helper to create a minimal quantized tensor for testing
// We just need the tensor to exist; we're testing kernel creation, not computation

TEST_F(Test__KernelFactory, CreateGemm_IQ4_NL_CPU)
{
    // Create minimal IQ4_NL tensor (32 rows x 32 cols)
    const size_t rows = 32;
    const size_t cols = 32;
    const size_t block_size = 32;
    const size_t bytes_per_block = 18; // IQ4_NL block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    IQ4_NLTensor tensor({rows, cols}, raw_data);

    // Create GEMM kernel via factory
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_Q4_0_CPU)
{
    const size_t rows = 32;
    const size_t cols = 32;
    const size_t block_size = 32;
    const size_t bytes_per_block = 18; // Q4_0 block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    Q4_0Tensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_Q4_1_CPU)
{
    const size_t rows = 32;
    const size_t cols = 32;
    const size_t block_size = 32;
    const size_t bytes_per_block = 20; // Q4_1 block size (has min value)
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    Q4_1Tensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_Q5_0_CPU)
{
    const size_t rows = 32;
    const size_t cols = 32;
    const size_t block_size = 32;
    const size_t bytes_per_block = 22; // Q5_0 block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    Q5_0Tensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_Q5_1_CPU)
{
    const size_t rows = 32;
    const size_t cols = 32;
    const size_t block_size = 32;
    const size_t bytes_per_block = 24; // Q5_1 block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    Q5_1Tensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_Q6_K_CPU)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 256;
    const size_t bytes_per_block = 210; // Q6_K block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    Q6_KTensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_Q8_0_CPU)
{
    const size_t rows = 32;
    const size_t cols = 32;
    const size_t block_size = 32;
    const size_t bytes_per_block = 34; // Q8_0 block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    Q8_0Tensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_Q8_1_CPU)
{
    const size_t rows = 32;
    const size_t cols = 32;
    const size_t block_size = 32;
    const size_t bytes_per_block = 36; // Q8_1 block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    Q8_1Tensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

// ============================================================================
// createGemm() Tests - K-Quant Tensor Types (CPU)
// ============================================================================

TEST_F(Test__KernelFactory, CreateGemm_Q2_K_CPU)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 256;
    const size_t bytes_per_block = 84; // Q2_K block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    Q2_KTensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_Q3_K_CPU)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 256;
    const size_t bytes_per_block = 110; // Q3_K block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    Q3_KTensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_Q4_K_CPU)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 256;
    const size_t bytes_per_block = 144; // Q4_K block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    Q4_KTensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_Q5_K_CPU)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 256;
    const size_t bytes_per_block = 176; // Q5_K block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    Q5_KTensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

// Q8_K does NOT implement IINT8Unpackable, so kernel creation should throw.
// This is intentional - Q8_K uses the tile-based ITensorGemmTileDataProvider path
// via TileBasedGemmKernel, not the packed GEMM path.
TEST_F(Test__KernelFactory, CreateGemm_Q8_K_CPU)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 256;
    const size_t bytes_per_block = 292; // Q8_K block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    Q8_KTensor tensor({rows, cols}, raw_data);

    // Q8_K now implements IINT8Unpackable, supporting packed GEMM via INT8 requantization
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

// ============================================================================
// createGemm() Tests - IQ (Intelligent Quant) Tensor Types (CPU)
// ============================================================================

TEST_F(Test__KernelFactory, CreateGemm_IQ1_M_CPU)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 256;
    const size_t bytes_per_block = 56; // IQ1_M block size (56 bytes)
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    IQ1_MTensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_IQ1_S_CPU)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 256;
    const size_t bytes_per_block = 50; // IQ1_S block size (50 bytes)
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    IQ1_STensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_IQ2_S_CPU)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 256;
    const size_t bytes_per_block = 82; // IQ2_S block size (82 bytes)
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    IQ2_STensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_IQ2_XS_CPU)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 256;
    const size_t bytes_per_block = 74; // IQ2_XS block size (74 bytes)
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    IQ2_XSTensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_IQ2_XXS_CPU)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 256;
    const size_t bytes_per_block = 66; // IQ2_XXS block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    IQ2_XXSTensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_IQ3_S_CPU)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 256;
    const size_t bytes_per_block = 110; // IQ3_S block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    IQ3_STensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_IQ3_XXS_CPU)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 256;
    const size_t bytes_per_block = 98; // IQ3_XXS block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    IQ3_XXSTensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_IQ4_XS_CPU)
{
    const size_t rows = 256;
    const size_t cols = 256;
    const size_t block_size = 256;
    const size_t bytes_per_block = 136; // IQ4_XS block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    IQ4_XSTensor tensor({rows, cols}, raw_data);
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

// ============================================================================
// createGemm() Tests - Floating Point Tensor Types (CPU)
// ============================================================================

TEST_F(Test__KernelFactory, CreateGemm_FP32_CPU)
{
    FP32Tensor tensor({32, 32});
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_FP16_CPU)
{
    FP16Tensor tensor({32, 32});
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, CreateGemm_BF16_CPU)
{
    BF16Tensor tensor({32, 32});
    auto kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactory, PrepareGemm_FP32TensorSlice_UsesFloatingPointKind)
{
    std::unique_ptr<TensorBase> inner = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 32});
    SliceMetadata metadata;
    metadata.mode = SliceMode::FULL;
    metadata.original_rows = 32;
    metadata.original_cols = 32;
    metadata.slice_start = 0;
    metadata.slice_end = 32;

    TensorSlice slice(std::move(inner), metadata);

    ASSERT_NE(dynamic_cast<const IINT8Unpackable *>(&slice), nullptr)
        << "TensorSlice inherits IINT8Unpackable even when the inner tensor is FP32";
    ASSERT_EQ(slice.vnniFormatInfo(), nullptr)
        << "FP32 TensorSlice must not be treated as VNNI-packable";

    auto prepared = KernelFactory::prepareGemmHandleLocal(&slice, DeviceId::cpu());

    ASSERT_NE(prepared, nullptr);
    EXPECT_EQ(prepared->kind, KernelFactory::GemmPreparationKind::FLOATING_POINT);
    ASSERT_NE(prepared->prepared_weights, nullptr);
    EXPECT_EQ(prepared->prepared_weights->kind, KernelFactory::GemmPreparationKind::FLOATING_POINT);
    EXPECT_NE(prepared->prepared_weights->kernel, nullptr);
}

// ============================================================================
// Error Handling Tests - Unsupported GPU Backends
// ============================================================================

TEST_F(Test__KernelFactory, CreateGemm_IQ4_NL_CUDA_Throws)
{
    // CUDA is not implemented for IQ4_NL yet (unless HAVE_CUDA is defined)
#ifndef HAVE_CUDA
    const size_t rows = 32;
    const size_t cols = 32;
    const size_t block_size = 32;
    const size_t bytes_per_block = 18;
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    IQ4_NLTensor tensor({rows, cols}, raw_data);

    EXPECT_THROW(
        KernelFactory::createGemm(&tensor, DeviceType::CUDA),
        std::runtime_error);
#else
    GTEST_SKIP() << "CUDA is enabled, cannot test unsupported path";
#endif
}

// NOTE: CreateGemm_Q4_0_ROCm test moved to integration tests
// See: tests/v2/integration/Test__ROCmQuantisedGemmKernel.cpp
// Unit tests don't have HAVE_ROCM enabled, so it would never run here.

TEST_F(Test__KernelFactory, CreateGemm_FP32_Vulkan_Throws)
{
    FP32Tensor tensor({32, 32});

    // Vulkan is not implemented yet
    EXPECT_THROW(
        KernelFactory::createGemm(&tensor, DeviceType::Vulkan),
        std::runtime_error);
}

TEST_F(Test__KernelFactory, CreateGemm_FP32_Metal_Throws)
{
    FP32Tensor tensor({32, 32});

    // Metal is not implemented yet
    EXPECT_THROW(
        KernelFactory::createGemm(&tensor, DeviceType::Metal),
        std::runtime_error);
}

// ============================================================================
// Consistency Tests - Kernel Created via Factory Matches Direct Creation
// ============================================================================

TEST_F(Test__KernelFactory, CreateGemm_ConsistentWithTensorCreateGemm)
{
    // Verify that factory-created kernel and tensor-created kernel are the same type
    const size_t rows = 32;
    const size_t cols = 32;
    const size_t block_size = 32;
    const size_t bytes_per_block = 18;
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    IQ4_NLTensor tensor({rows, cols}, raw_data);

    // Create via factory
    auto factory_kernel = KernelFactory::createGemm(&tensor, DeviceType::CPU);

    // Create via tensor (which now uses KernelFactory internally)
    auto tensor_kernel = tensor.createGemm();

    // Both should be non-null
    ASSERT_NE(factory_kernel, nullptr);
    ASSERT_NE(tensor_kernel, nullptr);

    // Both should be same type (CPUNativeVNNIGemmKernel for CPU)
    // We can't easily compare types, but we can verify they're both valid
}

// ============================================================================
// createGemmRaw() Tests (IQ4_NL only has raw variant)
// ============================================================================

TEST_F(Test__KernelFactory, CreateGemmRaw_IQ4_NL_CPU)
{
    const size_t rows = 32;
    const size_t cols = 32;
    const size_t block_size = 32;
    const size_t bytes_per_block = 18;
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    IQ4_NLTensor tensor({rows, cols}, raw_data);

    // Create raw pointer (caller owns)
    ITensorGemm *kernel = KernelFactory::createGemmRaw(&tensor, DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);

    // Clean up
    delete kernel;
}
