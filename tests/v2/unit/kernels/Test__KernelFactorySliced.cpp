/**
 * @file Test__KernelFactorySliced.cpp
 * @brief Unit tests for KernelFactory::getOrCreateGemmSliced()
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests the sliced GEMM kernel API for tensor parallelism (Phase 2.1).
 */

#include <gtest/gtest.h>
#include <random>
#include "kernels/KernelFactory.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include "tensors/Tensors.h"
#include "backends/ComputeBackend.h"

using namespace llaminar2;
using namespace llaminar2::gemm_v4;
using namespace llaminar::v2::kernels;

class KernelFactorySlicedTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Clear cache before each test
        KernelFactory::clearCache();
        // Ensure DeviceManager is initialized
        DeviceManager::instance();
    }

    void TearDown() override
    {
        // Clean up cache after each test
        KernelFactory::clearCache();
    }

    // Helper to create Q8_0 tensor with random data
    std::unique_ptr<Q8_0Tensor> createQ8_0(size_t rows, size_t cols)
    {
        const size_t block_size = 32;
        const size_t bytes_per_block = 34; // Q8_0 block: 2 bytes scale + 32 bytes data
        const size_t num_blocks = rows * (cols / block_size);
        std::vector<uint8_t> raw_data(num_blocks * bytes_per_block);

        // Fill with random data
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto &byte : raw_data)
        {
            byte = static_cast<uint8_t>(dist(gen));
        }
        // Set scale values to reasonable floats
        for (size_t b = 0; b < num_blocks; ++b)
        {
            uint16_t scale_bits = 0x3C00; // ~1.0 in FP16
            memcpy(&raw_data[b * bytes_per_block], &scale_bits, sizeof(scale_bits));
        }

        return std::make_unique<Q8_0Tensor>(std::vector<size_t>{rows, cols}, std::move(raw_data));
    }

    // Helper to create Q4_0 tensor with random data
    std::unique_ptr<Q4_0Tensor> createQ4_0(size_t rows, size_t cols)
    {
        const size_t block_size = 32;
        const size_t bytes_per_block = 18; // Q4_0 block: 2 bytes scale + 16 bytes data
        const size_t num_blocks = rows * (cols / block_size);
        std::vector<uint8_t> raw_data(num_blocks * bytes_per_block);

        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto &byte : raw_data)
        {
            byte = static_cast<uint8_t>(dist(gen));
        }
        for (size_t b = 0; b < num_blocks; ++b)
        {
            uint16_t scale_bits = 0x3C00;
            memcpy(&raw_data[b * bytes_per_block], &scale_bits, sizeof(scale_bits));
        }

        return std::make_unique<Q4_0Tensor>(std::vector<size_t>{rows, cols}, std::move(raw_data));
    }

    // Helper to create FP32 tensor
    std::unique_ptr<FP32Tensor> createFP32(size_t rows, size_t cols, float fill_value = 0.0f)
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < rows * cols; ++i)
        {
            data[i] = fill_value;
        }
        return tensor;
    }

    // Helper to create FP32 tensor with random data
    std::unique_ptr<FP32Tensor> createFP32Random(size_t rows, size_t cols)
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
        float *data = tensor->mutable_data();
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < rows * cols; ++i)
        {
            data[i] = dist(gen);
        }
        return tensor;
    }
};

static ITensorGemm *getPreparedKernel(const TensorBase *tensor, DeviceId device = DeviceId::cpu())
{
    auto *prepared = KernelFactory::getOrCreatePreparedGemmWeights(tensor, device);
    return KernelFactory::getOrCreateGemmEngine(prepared);
}

// =============================================================================
// Basic Sliced Kernel Creation
// =============================================================================

TEST_F(KernelFactorySlicedTest, CreatesSlicedKernel)
{
    // Create Q8_0 weight tensor [1024, 896]
    auto weights = createQ8_0(1024, 896);
    ASSERT_NE(weights, nullptr);

    // Create sliced kernel for first half [0, 512)
    auto *kernel = KernelFactory::getOrCreateGemmSliced(weights.get(), 0, 512);
    ASSERT_NE(kernel, nullptr);

    // Verify kernel dimensions via dynamic_cast to QuantisedGemmKernel
    auto *qgemm = dynamic_cast<QuantisedGemmKernel *>(kernel);
    ASSERT_NE(qgemm, nullptr);
    EXPECT_EQ(qgemm->get_n(), 512); // Sliced N
    EXPECT_EQ(qgemm->get_k(), 896); // K unchanged
}

TEST_F(KernelFactorySlicedTest, SlicedKernelSecondHalf)
{
    // Create weight tensor [1024, 896]
    auto weights = createQ8_0(1024, 896);
    ASSERT_NE(weights, nullptr);

    // Create sliced kernel for second half [512, 1024)
    auto *kernel = KernelFactory::getOrCreateGemmSliced(weights.get(), 512, 1024);
    ASSERT_NE(kernel, nullptr);

    auto *qgemm = dynamic_cast<QuantisedGemmKernel *>(kernel);
    ASSERT_NE(qgemm, nullptr);
    EXPECT_EQ(qgemm->get_n(), 512); // Sliced N
    EXPECT_EQ(qgemm->get_k(), 896); // K unchanged
}

// =============================================================================
// Cache Behavior
// =============================================================================

TEST_F(KernelFactorySlicedTest, CachesSlicedKernels)
{
    auto weights = createQ8_0(1024, 896);
    ASSERT_NE(weights, nullptr);

    // Create same sliced kernel twice
    auto *k1 = KernelFactory::getOrCreateGemmSliced(weights.get(), 0, 512);
    auto *k2 = KernelFactory::getOrCreateGemmSliced(weights.get(), 0, 512);

    // Should return same cached kernel
    EXPECT_EQ(k1, k2);
}

TEST_F(KernelFactorySlicedTest, DifferentSlicesGetDifferentKernels)
{
    auto weights = createQ8_0(1024, 896);
    ASSERT_NE(weights, nullptr);

    // Create different slices
    auto *k1 = KernelFactory::getOrCreateGemmSliced(weights.get(), 0, 512);
    auto *k2 = KernelFactory::getOrCreateGemmSliced(weights.get(), 512, 1024);

    // Should be different kernels
    EXPECT_NE(k1, k2);
}

TEST_F(KernelFactorySlicedTest, FullKernelDifferentFromSliced)
{
    auto weights = createQ8_0(1024, 896);
    ASSERT_NE(weights, nullptr);

    // Get full kernel
    auto *full = getPreparedKernel(weights.get());

    // Get sliced kernel covering full range
    auto *sliced = KernelFactory::getOrCreateGemmSliced(weights.get(), 0, 1024);

    // Should be different cache entries (different API paths)
    EXPECT_NE(full, sliced);

    // But both should have same dimensions
    auto *qfull = dynamic_cast<QuantisedGemmKernel *>(full);
    auto *qsliced = dynamic_cast<QuantisedGemmKernel *>(sliced);
    ASSERT_NE(qfull, nullptr);
    ASSERT_NE(qsliced, nullptr);
    EXPECT_EQ(qfull->get_n(), qsliced->get_n());
    EXPECT_EQ(qfull->get_k(), qsliced->get_k());
}

TEST_F(KernelFactorySlicedTest, ClearCacheForClearsSlicedEntries)
{
    auto weights = createQ8_0(1024, 896);
    ASSERT_NE(weights, nullptr);

    // Create sliced kernels
    auto *k1 = KernelFactory::getOrCreateGemmSliced(weights.get(), 0, 512);
    auto *k2 = KernelFactory::getOrCreateGemmSliced(weights.get(), 512, 1024);
    ASSERT_NE(k1, nullptr);
    ASSERT_NE(k2, nullptr);

    // Clear cache for this tensor
    KernelFactory::clearCacheFor(weights.get());

    // Get stats - should show 0 cached kernels
    auto [size, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size, 0);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(KernelFactorySlicedTest, NullTensorReturnsNull)
{
    auto *kernel = KernelFactory::getOrCreateGemmSliced(nullptr, 0, 512);
    EXPECT_EQ(kernel, nullptr);
}

TEST_F(KernelFactorySlicedTest, InvalidRangeThrows)
{
    auto weights = createQ8_0(1024, 896);
    ASSERT_NE(weights, nullptr);

    // start >= end
    EXPECT_THROW(
        KernelFactory::getOrCreateGemmSliced(weights.get(), 512, 512),
        std::runtime_error);

    EXPECT_THROW(
        KernelFactory::getOrCreateGemmSliced(weights.get(), 600, 500),
        std::runtime_error);
}

TEST_F(KernelFactorySlicedTest, RangeExceedsTensorThrows)
{
    auto weights = createQ8_0(1024, 896);
    ASSERT_NE(weights, nullptr);

    // row_end > tensor rows
    EXPECT_THROW(
        KernelFactory::getOrCreateGemmSliced(weights.get(), 0, 2048),
        std::runtime_error);
}

// =============================================================================
// Sliced GEMM Correctness
// =============================================================================

TEST_F(KernelFactorySlicedTest, SlicedGemmProducesCorrectOutput)
{
    // Small dimensions for easy verification
    // NOTE: QuantisedGemmKernel is optimized for M=1 (single token inference)
    // For tensor parallelism, row-slicing operates on the output dimension N
    const size_t N = 64; // Output rows (weight rows)
    const size_t K = 32; // Input cols (weight cols)
    const size_t M = 1;  // Single token - kernel's expected use case

    // Create weight tensor [N, K]
    auto weights = createQ8_0(N, K);
    ASSERT_NE(weights, nullptr);

    // Create input tensor [M, K]
    auto input = createFP32Random(M, K);
    ASSERT_NE(input, nullptr);

    // Create output tensors for full and sliced computation
    auto output_full = createFP32(M, N, 0.0f);
    auto output_sliced1 = createFP32(M, N / 2, 0.0f);
    auto output_sliced2 = createFP32(M, N / 2, 0.0f);

    // Compute full GEMM
    auto *full_kernel = getPreparedKernel(weights.get());
    ASSERT_NE(full_kernel, nullptr);
    bool ok = full_kernel->multiply(
        input->data(),
        output_full->mutable_data(),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), 1.0f, 0.0f);
    ASSERT_TRUE(ok);

    // Compute sliced GEMMs
    auto *sliced1 = KernelFactory::getOrCreateGemmSliced(weights.get(), 0, N / 2);
    auto *sliced2 = KernelFactory::getOrCreateGemmSliced(weights.get(), N / 2, N);
    ASSERT_NE(sliced1, nullptr);
    ASSERT_NE(sliced2, nullptr);

    ok = sliced1->multiply(
        input->data(),
        output_sliced1->mutable_data(),
        static_cast<int>(M), static_cast<int>(N / 2), static_cast<int>(K), 1.0f, 0.0f);
    ASSERT_TRUE(ok);

    ok = sliced2->multiply(
        input->data(),
        output_sliced2->mutable_data(),
        static_cast<int>(M), static_cast<int>(N / 2), static_cast<int>(K), 1.0f, 0.0f);
    ASSERT_TRUE(ok);

    // Compare: concatenation of sliced outputs should match full output
    const float *full_data = output_full->data();
    const float *slice1_data = output_sliced1->data();
    const float *slice2_data = output_sliced2->data();

    for (size_t m = 0; m < M; ++m)
    {
        // First half of row
        for (size_t n = 0; n < N / 2; ++n)
        {
            float expected = full_data[m * N + n];
            float actual = slice1_data[m * (N / 2) + n];
            EXPECT_NEAR(expected, actual, 1e-3f)
                << "Mismatch at row " << m << ", col " << n << " (first half)";
        }
        // Second half of row
        for (size_t n = 0; n < N / 2; ++n)
        {
            float expected = full_data[m * N + N / 2 + n];
            float actual = slice2_data[m * (N / 2) + n];
            EXPECT_NEAR(expected, actual, 1e-3f)
                << "Mismatch at row " << m << ", col " << (N / 2 + n) << " (second half)";
        }
    }
}

// =============================================================================
// Multiple Tensor Types
// =============================================================================

TEST_F(KernelFactorySlicedTest, WorksWithQ4_0)
{
    auto weights = createQ4_0(1024, 896);
    ASSERT_NE(weights, nullptr);

    auto *kernel = KernelFactory::getOrCreateGemmSliced(weights.get(), 0, 512);
    ASSERT_NE(kernel, nullptr);

    auto *qgemm = dynamic_cast<QuantisedGemmKernel *>(kernel);
    ASSERT_NE(qgemm, nullptr);
    EXPECT_EQ(qgemm->get_n(), 512);
}
