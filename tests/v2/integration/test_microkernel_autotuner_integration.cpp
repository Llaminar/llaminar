/**
 * @file test_microkernel_autotuner_integration.cpp
 * @brief Test microkernel registry integration with auto-tuner
 * 
 * Verifies that the auto-tuner can access all 1,225 pre-compiled
 * microkernel instantiations and select optimal configurations.
 * 
 * @author David Sanftenberg
 * @date October 2025
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <algorithm>
#include "kernels/cpu/GemmMicroKernelRegistry.h"
#include "kernels/cpu/GemmAutoTuner.h"
#include "kernels/cpu/GemmMicroKernelAdapter.h"
#include "tensors/TensorKernels.h"
#include <iostream>
#include <memory>
#include <cstring>

using namespace llaminar2;
using namespace llaminar::v2::kernels;

/**
 * @brief Mock block decoder for testing
 */
class TestBlockDecoder : public IBlockDecoder
{
public:
    TestBlockDecoder(size_t rows, size_t cols, size_t block_size = 32)
        : rows_(rows), cols_(cols), block_size_(block_size)
    {
        // Initialize with simple test pattern
        size_t blocks_per_row = (cols + block_size - 1) / block_size;
        data_.resize(rows * blocks_per_row * block_size, 1.0f);
    }

    void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
    {
        size_t blocks_per_row = (cols_ + block_size_ - 1) / block_size_;
        size_t offset = (row_idx * blocks_per_row + k_block_offset) * block_size_;
        std::memcpy(output, &data_[offset], block_size_ * sizeof(float));
    }

    const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
    {
        size_t blocks_per_row = (cols_ + block_size_ - 1) / block_size_;
        size_t offset = (row_idx * blocks_per_row + k_block_offset) * block_size_;
        return &data_[offset];
    }

    size_t decoder_rows() const override { return rows_; }
    size_t decoder_cols() const override { return cols_; }
    size_t block_size() const override { return block_size_; }

private:
    size_t rows_;
    size_t cols_;
    size_t block_size_;
    std::vector<float> data_;
};

/**
 * @brief Test: Verify registry has 1,225 kernels available
 */
TEST(MicroKernelAutoTunerIntegration, RegistryPopulated)
{
    auto &registry = kernels::gemm::MicroKernelRegistry::instance();
    
    EXPECT_EQ(registry.size(), 1225)
        << "Registry should contain 1,225 pre-compiled instantiations";
}

/**
 * @brief Test: Verify auto-tuner can access registered variants
 */
TEST(MicroKernelAutoTunerIntegration, VariantsRegistered)
{
    TestBlockDecoder decoder(896, 896); // Qwen 0.5B hidden size
    
    // Register all variants from microkernel registry
    auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
    
    std::cout << "Registered " << variants.size() << " variants with auto-tuner\n";
    
    // Should have 1,225 variants available
    EXPECT_EQ(variants.size(), 1225)
        << "Auto-tuner should receive all 1,225 variants from registry";
    
    // Verify at least some variants are present
    EXPECT_GT(variants.size(), 0) << "At least some variants should be registered";
}

/**
 * @brief Test: Auto-tuner can select optimal kernel for typical shapes
 */
TEST(MicroKernelAutoTunerIntegration, AutoTunerSelection)
{
    std::cout << "\n[VERIFICATION] Testing all shapes with padding fix\n" << std::flush;
    TestBlockDecoder decoder(896, 896);
    
    auto &tuner = GemmAutoTuner::instance();
    tuner.clearCache(); // Start fresh
    
    // Register variants
    auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
    for (auto &variant : variants)
    {
        tuner.registerVariant(std::move(variant));
    }
    
    // Test typical inference shapes - full performance matrix
    const std::vector<std::tuple<int, int, int>> test_shapes = {
        {1, 896, 896},    // Single token decode
        {8, 896, 896},    // Small batch
        {32, 896, 896},   // Medium batch
        {128, 896, 896},  // Large batch
        {512, 896, 896},  // Prefill
        {1024, 896, 896}, // Large prefill (was crashing before padding fix)
        {2048, 896, 896}, // Very large prefill
        {4096, 896, 896}, // Extreme prefill
    };
    
    for (const auto &[m, n, k] : test_shapes)
    {
        auto *kernel = tuner.getOptimalKernel(m, n, k);
        
        ASSERT_NE(kernel, nullptr)
            << "Auto-tuner should return valid kernel for shape ("
            << m << ", " << n << ", " << k << ")";
        
        auto config = kernel->config();
        std::cout << "Shape (" << m << ", " << n << ", " << k << ") -> "
                  << "tile " << config.tile_m << "×" << config.tile_n
                  << ", unroll=" << config.unroll_factor
                  << ", prefetch=" << config.prefetch_blocks << "\n";
    }
}

/**
 * @brief Test: Verify L1Opt configuration is available
 */
TEST(MicroKernelAutoTunerIntegration, L1OptConfigAvailable)
{
    auto &registry = kernels::gemm::MicroKernelRegistry::instance();
    
    // L1Opt config from analysis: AVX512, 8×6 tile, unroll=4, prefetch=2
    bool has_l1opt = registry.has_kernel("simd::AVX512Tag", 8, 6, 4, 2);
    
    EXPECT_TRUE(has_l1opt)
        << "L1Opt configuration (666 GFLOPS baseline) should be in registry";
    
    if (has_l1opt)
    {
        auto bundle = registry.get_kernel("simd::AVX512Tag", 8, 6, 4, 2);
        EXPECT_NE(bundle.micro_kernel, nullptr)
            << "L1Opt kernel function should be valid";
    }
}

/**
 * @brief Test: Kernel execution produces correct results
 */
TEST(MicroKernelAutoTunerIntegration, KernelCorrectnessSmoke)
{
    // Small smoke test: 4×4 matrix multiplication
    const int m = 4, n = 4, k = 4;
    
    TestBlockDecoder decoder(k, n);
    
    auto &tuner = GemmAutoTuner::instance();
    tuner.clearCache();
    
    // Register variants
    auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
    for (auto &variant : variants)
    {
        tuner.registerVariant(std::move(variant));
    }
    
    // Get kernel
    auto *kernel = tuner.getOptimalKernel(m, n, k);
    ASSERT_NE(kernel, nullptr);
    
    // Create test matrices
    std::vector<float> A(m * k, 1.0f); // All ones
    std::vector<float> C(m * n, 0.0f); // Initialize to zero
    
    // Execute kernel: C = A × B (B is all ones from decoder)
    bool success = kernel->multiply(
        A.data(), C.data(), m, n, k, &decoder, 1.0f, 0.0f);
    
    EXPECT_TRUE(success) << "Kernel execution should succeed";
    
    // Verify result: each element should be k (sum of k ones)
    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(C[i], static_cast<float>(k), 1e-3f)
            << "C[" << i << "] should equal k=" << k;
    }
}

/**
 * Test the problematic 1024×896×896 shape that crashes
 */
TEST(MicroKernelAutoTunerIntegration, LargeShape1024x896x896)
{
    const int m = 1024, n = 896, k = 896;
    
    std::cout << "\n[TEST] Testing problematic shape: m=" << m << ", n=" << n << ", k=" << k << "\n";
    std::cout << "[TEST] Buffer sizes: A=" << (m*k*4) << " bytes, C=" << (m*n*4) << " bytes\n";
    
    TestBlockDecoder decoder(k, n);
    
    auto &tuner = GemmAutoTuner::instance();
    tuner.clearCache();
    
    // Register variants
    auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
    for (auto &variant : variants)
    {
        tuner.registerVariant(std::move(variant));
    }
    
    std::cout << "[TEST] About to call getOptimalKernel() - this is where it crashes\n" << std::flush;
    
    // Get kernel - THIS IS WHERE IT CRASHES
    auto *kernel = tuner.getOptimalKernel(m, n, k);
    
    std::cout << "[TEST] getOptimalKernel() returned successfully\n";
    ASSERT_NE(kernel, nullptr);
    
    // Create test matrices
    std::vector<float> A(m * k, 1.0f);
    std::vector<float> C(m * n, 0.0f);
    
    std::cout << "[TEST] About to execute kernel\n" << std::flush;
    
    // Execute kernel
    bool success = kernel->multiply(
        A.data(), C.data(), m, n, k, &decoder, 1.0f, 0.0f);
    
    std::cout << "[TEST] Kernel execution completed\n";
    EXPECT_TRUE(success) << "Kernel execution should succeed";
    
    // Verify result (spot check - full verification would be slow)
    EXPECT_NEAR(C[0], static_cast<float>(k), 1e-3f) << "C[0] should equal k=" << k;
    EXPECT_NEAR(C[m*n-1], static_cast<float>(k), 1e-3f) << "C[last] should equal k=" << k;
    
    std::cout << "[TEST] Test completed successfully\n";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
