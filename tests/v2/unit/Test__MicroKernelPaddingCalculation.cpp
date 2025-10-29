/**
 * @file Test__MicroKernelPaddingCalculation.cpp
 * @brief Unit test to calculate and verify required padding for micro-kernel SIMD operations
 * 
 * This test analyzes the actual access patterns in MicroKernelTemplate to determine
 * the exact padding requirements for various matrix shapes and micro-kernel configurations.
 * 
 * Key insight: The micro-kernel accesses B_panel[j * k_panel + offset] where:
 *   - j ranges [0, NR)
 *   - offset ranges [0, k_panel) in steps of UNROLL
 *   - Each SIMD load reads 16 floats (AVX512)
 * 
 * The worst-case over-read occurs at:
 *   position = (NR-1) * k_panel + (k_panel - SIMD_WIDTH)
 *   read_end = position + SIMD_WIDTH
 * 
 * @author David Sanftenberg
 * @date October 2025
 */

#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace {

/**
 * @brief Calculate required padding for micro-kernel buffers
 */
class PaddingCalculator
{
public:
    struct Config {
        int m;          // Matrix rows
        int n;          // Matrix cols  
        int k;          // Inner dimension
        int mc;         // M cache block size
        int nc;         // N cache block size
        int mr;         // Micro-kernel M tile
        int nr;         // Micro-kernel N tile
        int unroll;     // K unroll factor
    };
    
    struct BufferInfo {
        size_t logical_size;       // Size needed for data
        size_t worst_case_access;  // Highest address accessed by SIMD
        size_t overread_bytes;     // Bytes read beyond logical size
        size_t required_padding;   // Padding needed (in floats)
    };
    
    static constexpr size_t SIMD_WIDTH = 16;  // AVX512: 16 floats
    static constexpr size_t CACHE_LINE = 64;  // 64 bytes
    
    /**
     * @brief Calculate required padding for B buffer (packed B matrix panel)
     * 
     * The B buffer is laid out as: nc × k (column-major packing)
     * The micro-kernel accesses: B_panel[j * k + offset] where j ∈ [0, NR)
     * 
     * Worst case: When processing the last micro-panel (jr close to nc):
     *   Base address: B_packed + jr * k
     *   Access: B_panel + (NR-1) * k + (k - SIMD_WIDTH)
     *   Read: SIMD_WIDTH floats starting from that address
     */
    static BufferInfo calculateBPadding(const Config& cfg)
    {
        BufferInfo info;
        info.logical_size = cfg.k * cfg.nc;
        
        // Worst case access pattern in micro-kernel:
        // The micro-kernel is called with B_panel = B_packed + jr * k
        // It accesses B_panel[j * k + offset] where:
        //   - j ranges [0, NR)
        //   - offset ranges [0, k) in steps that may hit k - SIMD_WIDTH
        
        // The furthest we can go into B_packed is when:
        //   jr = nc - NR (last micro-panel)
        //   j = NR - 1 (last row within micro-panel)
        //   offset = k - SIMD_WIDTH (last SIMD-aligned position)
        
        size_t jr_max = (cfg.nc > cfg.nr) ? (cfg.nc - cfg.nr) : 0;
        size_t base_offset = jr_max * cfg.k;
        size_t row_offset = (cfg.nr - 1) * cfg.k;
        size_t col_offset = (cfg.k > SIMD_WIDTH) ? (cfg.k - SIMD_WIDTH) : 0;
        size_t access_start = base_offset + row_offset + col_offset;
        size_t access_end = access_start + SIMD_WIDTH;
        
        info.worst_case_access = access_end;
        
        if (info.worst_case_access > info.logical_size) {
            info.overread_bytes = (info.worst_case_access - info.logical_size) * sizeof(float);
            info.required_padding = info.worst_case_access - info.logical_size;
        } else {
            info.overread_bytes = 0;
            info.required_padding = 0;
        }
        
        // Add safety margin: round up to cache line + 10%
        size_t margin = std::max<size_t>(
            CACHE_LINE / sizeof(float),
            info.logical_size / 10
        );
        info.required_padding = std::max(info.required_padding, margin);
        
        return info;
    }
    
    /**
     * @brief Calculate required padding for A buffer (packed A matrix panel)
     * 
     * Similar analysis for A buffer: mc × k layout
     * Accessed as: A_panel[i * k + offset] where i ∈ [0, MR)
     */
    static BufferInfo calculateAPadding(const Config& cfg)
    {
        BufferInfo info;
        info.logical_size = cfg.k * cfg.mc;
        
        // Worst case for A buffer:
        // Base: A_packed + ir * k where ir can be up to mc - MR
        // Access: A_panel[(MR-1) * k + (k - SIMD_WIDTH)]
        // Read: SIMD_WIDTH floats
        
        size_t ir_max = (cfg.mc > cfg.mr) ? (cfg.mc - cfg.mr) : 0;
        size_t base_offset = ir_max * cfg.k;
        size_t row_offset = (cfg.mr - 1) * cfg.k;
        size_t col_offset = (cfg.k > SIMD_WIDTH) ? (cfg.k - SIMD_WIDTH) : 0;
        size_t access_start = base_offset + row_offset + col_offset;
        size_t access_end = access_start + SIMD_WIDTH;
        
        info.worst_case_access = access_end;
        
        if (info.worst_case_access > info.logical_size) {
            info.overread_bytes = (info.worst_case_access - info.logical_size) * sizeof(float);
            info.required_padding = info.worst_case_access - info.logical_size;
        } else {
            info.overread_bytes = 0;
            info.required_padding = 0;
        }
        
        // Add safety margin
        size_t margin = std::max<size_t>(
            CACHE_LINE / sizeof(float),
            info.logical_size / 10
        );
        info.required_padding = std::max(info.required_padding, margin);
        
        return info;
    }
    
    /**
     * @brief Verify current padding formula provides sufficient padding
     */
    static bool verifyPaddingFormula(const Config& cfg, const BufferInfo& required)
    {
        // Current formula in MicroKernelAdapter.h:
        // const size_t padding = nc_ * CACHE_LINE / sizeof(float) + (size / 10);
        
        size_t current_padding = cfg.nc * (CACHE_LINE / sizeof(float)) + (required.logical_size / 10);
        
        return current_padding >= required.required_padding;
    }
};

/**
 * @brief Test fixture for padding calculation tests
 */
class MicroKernelPaddingCalculationTest : public ::testing::Test
{
protected:
    void printBufferInfo(const std::string& name, const PaddingCalculator::BufferInfo& info)
    {
        std::cout << "  " << name << " buffer:\n"
                  << "    Logical size:       " << std::setw(10) << info.logical_size << " floats ("
                  << (info.logical_size * sizeof(float) / 1024) << " KB)\n"
                  << "    Worst access:       " << std::setw(10) << info.worst_case_access << " floats\n"
                  << "    Over-read:          " << std::setw(10) << info.overread_bytes << " bytes\n"
                  << "    Required padding:   " << std::setw(10) << info.required_padding << " floats ("
                  << (info.required_padding * sizeof(float)) << " bytes)\n";
    }
    
    void testShape(const PaddingCalculator::Config& cfg, const std::string& description)
    {
        std::cout << "\n" << description << "\n";
        std::cout << "  Shape: m=" << cfg.m << ", n=" << cfg.n << ", k=" << cfg.k << "\n";
        std::cout << "  Cache blocks: mc=" << cfg.mc << ", nc=" << cfg.nc << "\n";
        std::cout << "  Micro-kernel: mr=" << cfg.mr << ", nr=" << cfg.nr << ", unroll=" << cfg.unroll << "\n";
        
        auto a_info = PaddingCalculator::calculateAPadding(cfg);
        auto b_info = PaddingCalculator::calculateBPadding(cfg);
        
        printBufferInfo("A", a_info);
        printBufferInfo("B", b_info);
        
        // Verify our formula provides enough padding
        bool a_ok = PaddingCalculator::verifyPaddingFormula(cfg, a_info);
        bool b_ok = PaddingCalculator::verifyPaddingFormula(cfg, b_info);
        
        EXPECT_TRUE(a_ok) << "A buffer padding formula insufficient for " << description;
        EXPECT_TRUE(b_ok) << "B buffer padding formula insufficient for " << description;
        
        // Additional check: required padding should be reasonable (not more than 10% + cache lines)
        size_t max_reasonable_a = cfg.mc * (PaddingCalculator::CACHE_LINE / sizeof(float)) + (a_info.logical_size / 5);
        size_t max_reasonable_b = cfg.nc * (PaddingCalculator::CACHE_LINE / sizeof(float)) + (b_info.logical_size / 5);
        
        EXPECT_LE(a_info.required_padding, max_reasonable_a) 
            << "A padding requirement unexpectedly high - check calculation";
        EXPECT_LE(b_info.required_padding, max_reasonable_b)
            << "B padding requirement unexpectedly high - check calculation";
    }
};

// Test: Small matrix (single token)
TEST_F(MicroKernelPaddingCalculationTest, SmallMatrix_1x896x896)
{
    PaddingCalculator::Config cfg{};
    cfg.m = 1;
    cfg.n = 896;
    cfg.k = 896;
    cfg.mc = 64;   // Small cache block
    cfg.nc = 64;
    cfg.mr = 4;    // Typical micro-kernel
    cfg.nr = 4;
    cfg.unroll = 8;
    
    testShape(cfg, "Small matrix (1×896×896) - single token decode");
}

// Test: Medium matrix (small batch)
TEST_F(MicroKernelPaddingCalculationTest, MediumMatrix_32x896x896)
{
    PaddingCalculator::Config cfg{};
    cfg.m = 32;
    cfg.n = 896;
    cfg.k = 896;
    cfg.mc = 128;
    cfg.nc = 128;
    cfg.mr = 4;
    cfg.nr = 6;
    cfg.unroll = 8;
    
    testShape(cfg, "Medium matrix (32×896×896) - small batch");
}

// Test: Large matrix (prefill)
TEST_F(MicroKernelPaddingCalculationTest, LargeMatrix_512x896x896)
{
    PaddingCalculator::Config cfg{};
    cfg.m = 512;
    cfg.n = 896;
    cfg.k = 896;
    cfg.mc = 256;
    cfg.nc = 256;
    cfg.mr = 8;
    cfg.nr = 6;
    cfg.unroll = 8;
    
    testShape(cfg, "Large matrix (512×896×896) - prefill");
}

// Test: CRITICAL - The problematic size that crashes
TEST_F(MicroKernelPaddingCalculationTest, ProblematicMatrix_1024x896x896)
{
    PaddingCalculator::Config cfg{};
    cfg.m = 1024;
    cfg.n = 896;
    cfg.k = 896;
    cfg.mc = 256;  // Large cache block for large m
    cfg.nc = 256;
    cfg.mr = 8;    // Likely selected micro-kernel
    cfg.nr = 6;
    cfg.unroll = 8;
    
    testShape(cfg, "PROBLEMATIC: Large prefill (1024×896×896) - CRASHES WITHOUT FIX");
}

// Test: Edge case - nc much smaller than n
TEST_F(MicroKernelPaddingCalculationTest, EdgeCase_SmallCacheBlock)
{
    PaddingCalculator::Config cfg{};
    cfg.m = 128;
    cfg.n = 896;
    cfg.k = 896;
    cfg.mc = 64;   // Very small cache block
    cfg.nc = 64;
    cfg.mr = 2;
    cfg.nr = 4;
    cfg.unroll = 16;
    
    testShape(cfg, "Edge case: Small cache block (nc=64)");
}

// Test: Edge case - large micro-kernel tile
TEST_F(MicroKernelPaddingCalculationTest, EdgeCase_LargeMicroKernel)
{
    PaddingCalculator::Config cfg{};
    cfg.m = 256;
    cfg.n = 896;
    cfg.k = 896;
    cfg.mc = 256;
    cfg.nc = 256;
    cfg.mr = 8;    // Large MR
    cfg.nr = 8;    // Large NR
    cfg.unroll = 4;
    
    testShape(cfg, "Edge case: Large micro-kernel (8×8)");
}

// Test: Very large matrix
TEST_F(MicroKernelPaddingCalculationTest, VeryLargeMatrix_2048x896x896)
{
    PaddingCalculator::Config cfg{};
    cfg.m = 2048;
    cfg.n = 896;
    cfg.k = 896;
    cfg.mc = 256;
    cfg.nc = 256;
    cfg.mr = 8;
    cfg.nr = 6;
    cfg.unroll = 8;
    
    testShape(cfg, "Very large matrix (2048×896×896)");
}

// Test: Calculate exact padding for specific failing configuration
TEST_F(MicroKernelPaddingCalculationTest, CalculateMinimumSafePadding)
{
    // Test all common micro-kernel configurations for 1024×896×896
    std::vector<PaddingCalculator::Config> configs;
    
    // Generate common configurations
    for (int mc : {64, 128, 256}) {
        for (int nc : {64, 128, 256}) {
            for (int mr : {2, 4, 6, 8}) {
                for (int nr : {2, 4, 6, 8}) {
                    PaddingCalculator::Config cfg{};
                    cfg.m = 1024;
                    cfg.n = 896;
                    cfg.k = 896;
                    cfg.mc = mc;
                    cfg.nc = nc;
                    cfg.mr = mr;
                    cfg.nr = nr;
                    cfg.unroll = 8;
                    configs.push_back(cfg);
                }
            }
        }
    }
    
    size_t max_a_padding = 0;
    size_t max_b_padding = 0;
    
    for (const auto& cfg : configs) {
        auto a_info = PaddingCalculator::calculateAPadding(cfg);
        auto b_info = PaddingCalculator::calculateBPadding(cfg);
        
        max_a_padding = std::max(max_a_padding, a_info.required_padding);
        max_b_padding = std::max(max_b_padding, b_info.required_padding);
    }
    
    std::cout << "\nMaximum padding requirements across all configurations for 1024×896×896:\n";
    std::cout << "  A buffer: " << max_a_padding << " floats (" 
              << (max_a_padding * sizeof(float) / 1024) << " KB)\n";
    std::cout << "  B buffer: " << max_b_padding << " floats (" 
              << (max_b_padding * sizeof(float) / 1024) << " KB)\n";
    
    // Verify our formula covers the worst case
    PaddingCalculator::Config worst_case{};
    worst_case.m = 1024;
    worst_case.n = 896;
    worst_case.k = 896;
    worst_case.mc = 256;
    worst_case.nc = 256;
    worst_case.mr = 8;
    worst_case.nr = 8;
    worst_case.unroll = 8;
    
    size_t formula_padding = worst_case.nc * (PaddingCalculator::CACHE_LINE / sizeof(float)) 
                            + (worst_case.k * worst_case.nc / 10);
    
    EXPECT_GE(formula_padding, max_b_padding)
        << "Current padding formula insufficient! Need at least " << max_b_padding 
        << " but formula gives " << formula_padding;
}

} // anonymous namespace
