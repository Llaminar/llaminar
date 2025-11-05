/**
 * @file Test__Phase6_Int8_Minimal.cpp
 * @brief Minimal compilation test for Phase 6 Int8 DP4A kernel
 *
 * @author David Sanftenberg
 * @date November 5, 2025
 */

#include <gtest/gtest.h>
#include "kernels/cuda/CudaGemmJITPhase6.h"
#include "kernels/cuda/CudaGemmConfigPhase6.h"

using namespace llaminar2;
using namespace llaminar2::cuda;

TEST(Phase6Int8Minimal, KernelCompiles)
{
    // Just check that we can compile the kernel
    auto config = get_default_phase6_config();
    EXPECT_TRUE(config.is_valid());

    std::cout << "Phase 6 Config: tile_m=" << config.tile_m
              << ", tile_n=" << config.tile_n
              << ", tile_k=" << config.tile_k
              << ", threads=" << config.threads_per_block << "\n";

    // Try to compile
    try
    {
        std::cout << "Compiling Phase 6 kernel...\n";
        auto kernel = CudaGemmJITPhase6::compile(config);
        std::cout << "✓ Kernel compilation successful!\n";

        std::cout << "\n========================================\n";
        std::cout << "Phase 6 Int8 DP4A Kernel READY\n";
        std::cout << "========================================\n";
        std::cout << "Expected performance: 50-90 TFLOPS\n";
        std::cout << "Baseline (Phase 5): 17.5 TFLOPS\n";
        std::cout << "Expected speedup: 2.9-5.1×\n";
        std::cout << "========================================\n";
    }
    catch (const std::exception &e)
    {
        FAIL() << "Kernel compilation failed: " << e.what();
    }
}
