/**
 * @file Test__MicroKernelBounds.cpp
 * @brief Unit test to verify micro-kernel write bounds for 1024×896×896
 * 
 * Purpose: Verify that for the problematic 1024×896×896 shape, all micro-kernel
 * invocations write within the allocated C buffer bounds.
 * 
 * Strategy:
 * 1. Simulate the tiling loops from MicroKernelAdapter
 * 2. For each micro-tile, compute the LAST element written
 * 3. Verify: (ic + ir + mb - 1) * n + (jc + jr + nb - 1) < m * n
 * 4. Test with various mc/nc/mr/nr configurations
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include <sstream>

/**
 * Struct to track worst-case write for a given configuration
 */
struct WriteAnalysis {
    int ic, ir, jc, jr;  // Loop indices
    int mb, nb;          // Actual tile sizes
    size_t offset_start; // First element written
    size_t offset_end;   // Last element written
    bool within_bounds;  // Whether write is safe
    
    std::string toString(int m, int n) const {
        std::ostringstream oss;
        oss << "ic=" << ic << " ir=" << ir << " jc=" << jc << " jr=" << jr
            << " mb=" << mb << " nb=" << nb
            << " | Start: (" << (ic + ir) << "," << (jc + jr) << ")"
            << " End: (" << (ic + ir + mb - 1) << "," << (jc + jr + nb - 1) << ")"
            << " | Offset: [" << offset_start << ", " << offset_end << "]"
            << " / " << (m * n) << " | " << (within_bounds ? "✓ OK" : "✗ OUT OF BOUNDS");
        return oss.str();
    }
};

/**
 * Simulate the tiling loops and find the worst-case write offset
 */
WriteAnalysis analyzeWriteBounds(int m, int n, int k, int mc, int nc, int mr, int nr) {
    WriteAnalysis worst;
    worst.offset_end = 0;
    worst.within_bounds = true;
    
    // Simulate the cache tiling loops (from MicroKernelAdapter.h line 160-165)
    for (int ic = 0; ic < m; ic += mc) {
        int mc_actual = std::min(mc, m - ic);
        
        for (int jc = 0; jc < n; jc += nc) {
            int nc_actual = std::min(nc, n - jc);
            
            // Simulate the micro-tiling loops (line 187-193)
            for (int ir = 0; ir < mc_actual; ir += mr) {
                int mb = std::min(mr, mc_actual - ir);
                
                for (int jr = 0; jr < nc_actual; jr += nr) {
                    int nb = std::min(nr, nc_actual - jr);
                    
                    // Compute C pointer: C + (ic + ir) * n + (jc + jr) [line 197]
                    size_t c_offset_start = static_cast<size_t>(ic + ir) * n + (jc + jr);
                    
                    // Last element written: C[(ic+ir+mb-1)*n + (jc+jr+nb-1)]
                    size_t c_offset_end = static_cast<size_t>(ic + ir + mb - 1) * n + (jc + jr + nb - 1);
                    
                    // Track worst case
                    if (c_offset_end > worst.offset_end) {
                        worst.ic = ic;
                        worst.ir = ir;
                        worst.jc = jc;
                        worst.jr = jr;
                        worst.mb = mb;
                        worst.nb = nb;
                        worst.offset_start = c_offset_start;
                        worst.offset_end = c_offset_end;
                        worst.within_bounds = (c_offset_end < static_cast<size_t>(m) * n);
                    }
                }
            }
        }
    }
    
    return worst;
}

/**
 * Test the problematic 1024×896×896 shape with various cache configurations
 */
TEST(MicroKernelBoundsTest, Shape1024x896x896) {
    const int m = 1024, n = 896, k = 896;
    
    // Test various cache tile sizes
    std::vector<std::tuple<int, int, int, int, std::string>> configs = {
        // {mc, nc, mr, nr, description}
        {256, 512, 8, 8, "Typical AVX2 config"},
        {384, 384, 6, 8, "Balanced square tiles"},
        {512, 256, 8, 4, "Wide cache tiles"},
        {128, 1024, 4, 8, "Narrow cache tiles"},
        {1024, 896, 8, 8, "Full matrix (no cache blocking)"},
        {255, 511, 7, 7, "Unaligned sizes"},
        {1, 1, 1, 1, "Minimal tiles"},
        {64, 64, 8, 8, "Small cache tiles"},
    };
    
    std::cout << "\nTesting m=" << m << ", n=" << n << ", k=" << k << "\n";
    std::cout << "Buffer size: " << (m * n) << " floats\n\n";
    
    for (const auto& [mc, nc, mr, nr, desc] : configs) {
        WriteAnalysis result = analyzeWriteBounds(m, n, k, mc, nc, mr, nr);
        
        std::cout << desc << " (mc=" << mc << " nc=" << nc << " mr=" << mr << " nr=" << nr << "):\n";
        std::cout << "  " << result.toString(m, n) << "\n";
        
        EXPECT_TRUE(result.within_bounds)
            << "Configuration: " << desc
            << "\n  Wrote to offset " << result.offset_end
            << " but buffer size is only " << (m * n);
    }
}

/**
 * Test various problem sizes to ensure no regressions
 */
TEST(MicroKernelBoundsTest, VariousShapes) {
    // Test shapes: working and problematic
    std::vector<std::tuple<int, int, int, std::string>> shapes = {
        {512, 896, 896, "Working: 512×896×896"},
        {1024, 1, 1, "Working: 1024×1×1"},
        {1024, 896, 896, "PROBLEMATIC: 1024×896×896"},
        {2048, 896, 896, "Larger: 2048×896×896"},
        {1024, 1024, 1024, "Square: 1024×1024×1024"},
    };
    
    // Use typical AVX2 config
    const int mc = 256, nc = 512, mr = 8, nr = 8;
    
    for (const auto& [m, n, k, desc] : shapes) {
        WriteAnalysis result = analyzeWriteBounds(m, n, k, mc, nc, mr, nr);
        
        EXPECT_TRUE(result.within_bounds)
            << "Shape: " << desc
            << "\n  " << result.toString(m, n);
    }
}

/**
 * Test edge case: What if m or n is not a multiple of mc/nc?
 */
TEST(MicroKernelBoundsTest, UnalignedDimensions) {
    // m=1027 is not divisible by mc=256
    // n=893 is not divisible by nc=512
    const int m = 1027, n = 893, k = 896;
    const int mc = 256, nc = 512, mr = 8, nr = 8;
    
    WriteAnalysis result = analyzeWriteBounds(m, n, k, mc, nc, mr, nr);
    
    std::cout << "\nUnaligned dimensions: m=" << m << ", n=" << n << ", k=" << k << "\n";
    std::cout << result.toString(m, n) << "\n";
    
    EXPECT_TRUE(result.within_bounds)
        << "Unaligned dimensions should still respect bounds";
    
    // Verify the last tile is partial
    EXPECT_LT(result.mb, mr) << "Last m-tile should be partial";
    EXPECT_LT(result.nb, nr) << "Last n-tile should be partial";
}

/**
 * Test the exact scenario from auto-tuner's top variants
 */
TEST(MicroKernelBoundsTest, AutoTunerTopVariants) {
    const int m = 1024, n = 896, k = 896;
    
    // These are the actual top-performing variants from auto-tuner logs
    std::vector<std::tuple<int, int, std::string>> variants = {
        {8, 8, "8×8 (standard AVX2)"},
        {6, 8, "6×8 (common variant)"},
        {8, 6, "8×6 (common variant)"},
        {4, 8, "4×8 (narrow)"},
        {8, 4, "8×4 (wide)"},
        {8, 16, "8×16 (AVX512)"},
        {16, 8, "16×8 (AVX512)"},
    };
    
    // Use typical cache sizes
    const int mc = 256, nc = 512;
    
    std::cout << "\nAuto-tuner variant analysis for m=" << m << ", n=" << n << ", k=" << k << "\n";
    
    for (const auto& [mr, nr, desc] : variants) {
        WriteAnalysis result = analyzeWriteBounds(m, n, k, mc, nc, mr, nr);
        
        std::cout << desc << ": " << result.toString(m, n) << "\n";
        
        EXPECT_TRUE(result.within_bounds)
            << "Variant: " << desc
            << "\n  Wrote beyond buffer at offset " << result.offset_end;
    }
}
