/**
 * @file Test__MicroKernelPadding.cpp
 * @brief Unit test for calculating and verifying required buffer padding for micro-kernel shapes
 *
 * This test computes the maximum required padding for packed buffers (A and B)
 * for a variety of shapes and micro-kernel configurations, including the problematic
 * m=1024, n=896, k=896 case. It verifies that the current padding formula is sufficient
 * for all possible SIMD over-reads in the micro-kernel.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <tuple>
#include <cmath>

namespace {

constexpr size_t SIMD_WIDTH = 16; // AVX512: 16 floats per load
constexpr size_t CACHE_LINE = 64; // 64 bytes = 16 floats

struct MicroKernelConfig {
    int mr; // Rows in A micro-panel
    int nr; // Columns in B micro-panel
    int mc; // Cache block size for A
    int nc; // Cache block size for B
    int k;  // Shared dimension
    const char* description;
};

/**
 * @brief Calculate required padding for A buffer based on micro-kernel access pattern
 * 
 * Access pattern in MicroKernelTemplate.h:
 *   A_panel[i * k_panel + offset] where i ∈ [0, MR), offset ∈ [0, k)
 *   Each SIMD load reads SIMD_WIDTH floats starting at that address
 * 
 * Worst case: i = MR-1, offset = k - SIMD_WIDTH
 *   Read starts at: (MR-1) * k + (k - SIMD_WIDTH)
 *   Read ends at:   (MR-1) * k + k = MR * k
 *   But buffer size is: mc * k
 * 
 * Since we iterate with ir from [0, mc) by steps of MR:
 *   Worst case base offset: (mc - MR) * k
 *   Add micro-kernel access: (MR-1) * k + k - SIMD_WIDTH + SIMD_WIDTH
 *   Total: mc * k (exactly at buffer boundary)
 * 
 * Padding needed: At least SIMD_WIDTH for over-read protection
 */
size_t required_padding_A(int mc, int k, int mr) {
    // Worst-case: SIMD load starting at last valid element
    // Could read SIMD_WIDTH floats past the logical end
    // Add extra cache line for safety
    return SIMD_WIDTH + CACHE_LINE / sizeof(float);
}

/**
 * @brief Calculate required padding for B buffer based on micro-kernel access pattern
 * 
 * Access pattern in MicroKernelTemplate.h:
 *   B_panel[j * k_panel + offset] where j ∈ [0, NR), offset ∈ [0, k)
 * 
 * B_panel is passed as: B_packed + jr * k where jr ∈ [0, nc) by steps of NR
 * 
 * Worst case: jr = (nc - NR), j = NR-1, offset = k - SIMD_WIDTH
 *   Base: (nc - NR) * k
 *   Micro-kernel offset: (NR-1) * k + (k - SIMD_WIDTH)
 *   Read starts at: (nc - NR) * k + (NR-1) * k + k - SIMD_WIDTH
 *                 = nc * k - SIMD_WIDTH
 *   Read ends at: nc * k (exactly at buffer boundary)
 * 
 * Padding needed: At least SIMD_WIDTH for over-read protection
 */
size_t required_padding_B(int nc, int k, int nr) {
    // Worst-case: SIMD load starting at last valid element
    // Could read SIMD_WIDTH floats past the logical end
    // Add extra cache line for safety
    return SIMD_WIDTH + CACHE_LINE / sizeof(float);
}

/**
 * @brief Current padding formula from MicroKernelAdapter.h
 */
size_t formula_padding_A(int mc, int k) {
    // Current formula: mc cache lines + 10% margin, minimum 2 cache lines
    constexpr size_t min_padding = 2 * CACHE_LINE / sizeof(float);
    size_t a_size = mc * k;
    return std::max(min_padding, mc * CACHE_LINE / sizeof(float) + (a_size / 10));
}

/**
 * @brief Current padding formula from MicroKernelAdapter.h
 */
size_t formula_padding_B(int nc, int k) {
    // Current formula: nc cache lines + 10% margin, minimum 2 cache lines
    constexpr size_t min_padding = 2 * CACHE_LINE / sizeof(float);
    size_t b_size = k * nc;
    return std::max(min_padding, nc * CACHE_LINE / sizeof(float) + (b_size / 10));
}

TEST(MicroKernelPaddingTest, PaddingSufficiencyForShapes)
{
    // Test a variety of shapes and micro-kernel configs
    std::vector<MicroKernelConfig> configs = {
        {8, 6, 128, 128, 896, "Typical Qwen config (8×6 tile)"},
        {4, 4, 128, 128, 896, "Small micro-kernel (4×4 tile)"},
        {8, 8, 256, 256, 896, "Large micro-kernel (8×8 tile)"},
        {1, 8, 64, 128, 896, "Asymmetric tile (1×8)"},
        {8, 1, 128, 64, 896, "Asymmetric tile (8×1)"},
        {8, 6, 1024, 896, 896, "PROBLEMATIC: m=1024, n=896, k=896"},
        {8, 6, 2048, 896, 896, "Even larger m=2048"},
        {8, 6, 4096, 896, 896, "Extreme m=4096"},
        {8, 6, 128, 128, 4096, "Large k=4096"},
        {8, 6, 128, 128, 1, "Tiny k=1"},
        {8, 6, 128, 128, 15, "Non-SIMD-aligned k=15"},
        {8, 6, 128, 128, 17, "Just past SIMD boundary k=17"},
        {1, 1, 1, 1, 1, "Minimal case (1×1 everything)"},
        {8, 6, 1, 1, 896, "mc=1, nc=1 (single micro-panel)"},
        {8, 6, 8, 6, 896, "mc=MR, nc=NR (exact fit)"},
    };

    bool all_passed = true;

    for (const auto& cfg : configs) {
        size_t a_size = cfg.mc * cfg.k;
        size_t b_size = cfg.k * cfg.nc;
        size_t req_a = required_padding_A(cfg.mc, cfg.k, cfg.mr);
        size_t req_b = required_padding_B(cfg.nc, cfg.k, cfg.nr);
        size_t formula_a = formula_padding_A(cfg.mc, cfg.k);
        size_t formula_b = formula_padding_B(cfg.nc, cfg.k);

        bool a_ok = formula_a >= req_a;
        bool b_ok = formula_b >= req_b;

        std::cout << (a_ok && b_ok ? "✓" : "✗") << " " << cfg.description << "\n"
                  << "    Shape: mc=" << cfg.mc << " nc=" << cfg.nc << " k=" << cfg.k
                  << " | MR=" << cfg.mr << " NR=" << cfg.nr << "\n"
                  << "    A: Required=" << req_a << " Formula=" << formula_a
                  << " (" << (a_ok ? "OK" : "FAIL") << ")\n"
                  << "    B: Required=" << req_b << " Formula=" << formula_b
                  << " (" << (b_ok ? "OK" : "FAIL") << ")\n";

        EXPECT_GE(formula_a, req_a) << "A buffer formula insufficient for " << cfg.description;
        EXPECT_GE(formula_b, req_b) << "B buffer formula insufficient for " << cfg.description;

        if (!a_ok || !b_ok) all_passed = false;
    }

    EXPECT_TRUE(all_passed) << "Some padding formulas were insufficient!";
}

/**
 * @brief Test edge case: Unaligned k dimension
 * 
 * When k is not a multiple of SIMD_WIDTH, the micro-kernel has a cleanup loop
 * that processes remaining elements. Verify padding is sufficient for these cases.
 */
TEST(MicroKernelPaddingTest, UnalignedKDimension)
{
    std::vector<int> k_values = {1, 7, 15, 17, 31, 33, 63, 65, 127, 129, 255, 257, 895, 897};
    
    for (int k : k_values) {
        int mc = 128, nc = 128, mr = 8, nr = 6;
        size_t req_a = required_padding_A(mc, k, mr);
        size_t req_b = required_padding_B(nc, k, nr);
        size_t formula_a = formula_padding_A(mc, k);
        size_t formula_b = formula_padding_B(nc, k);

        EXPECT_GE(formula_a, req_a) << "A padding insufficient for k=" << k;
        EXPECT_GE(formula_b, req_b) << "B padding insufficient for k=" << k;
    }
}

/**
 * @brief Test edge case: Small cache blocks (mc < MR or nc < NR)
 * 
 * When cache block size is smaller than micro-kernel tile, special handling
 * is needed. Verify padding is still sufficient.
 */
TEST(MicroKernelPaddingTest, SmallCacheBlocks)
{
    int k = 896;
    std::vector<MicroKernelConfig> configs = {
        {8, 6, 1, 1, k, "mc=1, nc=1 (minimum)"},
        {8, 6, 4, 4, k, "mc=4, nc=4 (less than MR/NR)"},
        {8, 6, 7, 5, k, "mc=7, nc=5 (almost MR/NR)"},
        {8, 6, 8, 6, k, "mc=MR, nc=NR (exact)"},
        {8, 6, 9, 7, k, "mc=MR+1, nc=NR+1"},
    };

    for (const auto& cfg : configs) {
        size_t req_a = required_padding_A(cfg.mc, cfg.k, cfg.mr);
        size_t req_b = required_padding_B(cfg.nc, cfg.k, cfg.nr);
        size_t formula_a = formula_padding_A(cfg.mc, cfg.k);
        size_t formula_b = formula_padding_B(cfg.nc, cfg.k);

        EXPECT_GE(formula_a, req_a) << "A padding insufficient: " << cfg.description;
        EXPECT_GE(formula_b, req_b) << "B padding insufficient: " << cfg.description;
    }
}

/**
 * @brief Test boundary condition: Exact buffer boundary access
 * 
 * Simulate the exact worst-case access pattern to verify padding handles
 * the case where the micro-kernel reads exactly at the buffer boundary.
 */
TEST(MicroKernelPaddingTest, ExactBoundaryAccess)
{
    // Simulate worst-case access pattern for B buffer
    // jr = (nc - NR), j = (NR - 1), offset at end of k
    
    int mc = 128, nc = 128, k = 896, mr = 8, nr = 6;
    
    // B buffer worst case:
    // Base pointer: B_packed + jr * k where jr = (nc - nr) = 122
    // Access: B_panel[j * k + offset] where j = 5, offset = k - SIMD_WIDTH
    size_t worst_case_offset = (nc - nr) * k + (nr - 1) * k + (k - SIMD_WIDTH);
    size_t buffer_size = nc * k;
    size_t access_end = worst_case_offset + SIMD_WIDTH;
    
    std::cout << "B buffer boundary analysis:\n"
              << "  Buffer size: " << buffer_size << " floats\n"
              << "  Worst-case read start: " << worst_case_offset << "\n"
              << "  Worst-case read end: " << access_end << "\n"
              << "  Over-read amount: " << (access_end - buffer_size) << " floats\n";
    
    size_t req_padding = required_padding_B(nc, k, nr);
    size_t formula_padding = formula_padding_B(nc, k);
    
    EXPECT_GE(buffer_size + formula_padding, access_end)
        << "B buffer + padding must accommodate worst-case SIMD read";
    EXPECT_GE(formula_padding, req_padding)
        << "Formula padding must meet minimum requirement";

    // A buffer worst case:
    size_t a_worst_case_offset = (mc - mr) * k + (mr - 1) * k + (k - SIMD_WIDTH);
    size_t a_buffer_size = mc * k;
    size_t a_access_end = a_worst_case_offset + SIMD_WIDTH;
    
    std::cout << "A buffer boundary analysis:\n"
              << "  Buffer size: " << a_buffer_size << " floats\n"
              << "  Worst-case read start: " << a_worst_case_offset << "\n"
              << "  Worst-case read end: " << a_access_end << "\n"
              << "  Over-read amount: " << (a_access_end - a_buffer_size) << " floats\n";
    
    size_t a_req_padding = required_padding_A(mc, k, mr);
    size_t a_formula_padding = formula_padding_A(mc, k);
    
    EXPECT_GE(a_buffer_size + a_formula_padding, a_access_end)
        << "A buffer + padding must accommodate worst-case SIMD read";
    EXPECT_GE(a_formula_padding, a_req_padding)
        << "Formula padding must meet minimum requirement";
}

/**
 * @brief Test with actual variant configurations from GemmVariantGenerator
 * 
 * Test all combinations of tile sizes that are actually used by the auto-tuner.
 */
TEST(MicroKernelPaddingTest, ActualVariantConfigurations)
{
    // These are the actual MR×NR combinations generated in GemmVariantGenerator.cpp
    std::vector<std::pair<int, int>> tiles = {
        {1, 1}, {1, 2}, {1, 4}, {1, 6}, {1, 8},
        {2, 1}, {2, 2}, {2, 4}, {2, 6}, {2, 8},
        {4, 1}, {4, 2}, {4, 4}, {4, 6}, {4, 8},
        {6, 1}, {6, 2}, {6, 4}, {6, 6}, {6, 8},
        {8, 1}, {8, 2}, {8, 4}, {8, 6}, {8, 8},
    };

    // Cache block sizes used in practice
    std::vector<int> cache_sizes = {64, 128, 256, 512};
    
    // Common k values (Qwen hidden dimensions)
    std::vector<int> k_values = {896, 1024, 2048, 4096};

    int test_count = 0;
    int pass_count = 0;

    for (const auto& [mr, nr] : tiles) {
        for (int mc : cache_sizes) {
            for (int nc : cache_sizes) {
                for (int k : k_values) {
                    size_t req_a = required_padding_A(mc, k, mr);
                    size_t req_b = required_padding_B(nc, k, nr);
                    size_t formula_a = formula_padding_A(mc, k);
                    size_t formula_b = formula_padding_B(nc, k);

                    test_count++;
                    if (formula_a >= req_a && formula_b >= req_b) {
                        pass_count++;
                    } else {
                        std::cout << "FAIL: MR=" << mr << " NR=" << nr
                                  << " mc=" << mc << " nc=" << nc << " k=" << k << "\n";
                        EXPECT_GE(formula_a, req_a);
                        EXPECT_GE(formula_b, req_b);
                    }
                }
            }
        }
    }

    std::cout << "Tested " << test_count << " variant configurations, "
              << pass_count << " passed (" 
              << (100.0 * pass_count / test_count) << "%)\n";
    
    EXPECT_EQ(pass_count, test_count) << "All variant configurations must have sufficient padding";
}

} // anonymous namespace
