/**
 * @file verify_registry.cpp
 * @brief Verify GemmMicroKernelRegistry contains all expected instantiations
 * @author David Sanftenberg
 */

#include "kernels/cpu/GemmMicroKernelRegistry.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace llaminar2::kernels::gemm;

int main() {
    auto& registry = MicroKernelRegistry::instance();
    
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ MicroKernel Registry Verification                         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";
    
    size_t total = registry.size();
    std::cout << "✅ Total registered kernels: " << total << "\n";
    std::cout << "   Expected: 1,225 instantiations\n";
    std::cout << "   Status: " << (total == 1225 ? "PASS ✅" : "FAIL ❌") << "\n\n";
    
    // Verify L1Opt configuration (666 GFLOPS baseline)
    std::cout << "🎯 Verifying L1Opt configuration (AVX512, 8×6, unroll=4, prefetch=2):\n";
    if (registry.has_kernel("simd::AVX512Tag", 8, 6, 4, 2)) {
        std::cout << "   ✅ L1Opt config FOUND in registry\n";
    } else {
        std::cout << "   ❌ L1Opt config MISSING\n";
    }
    
    // Verify auto-tuner required tiles
    std::cout << "\n🔍 Verifying auto-tuner required tiles:\n";
    std::vector<std::tuple<std::string, int, int, std::string>> required_tiles = {
        {"simd::AVX512Tag", 16, 64, "1024-token prefill"},
        {"simd::AVX512Tag", 16, 16, "512-token prefill"},
        {"simd::AVX512Tag", 2, 32, "128-token prefill"},
        {"simd::AVX512Tag", 4, 16, "32-token prefill"},
    };
    
    for (const auto& [isa, mr, nr, desc] : required_tiles) {
        bool found = false;
        // Check all unroll/prefetch combinations
        for (int unroll : {1, 2, 4, 8, 16}) {
            for (int prefetch : {0, 1, 2, 3, 5}) {
                if (registry.has_kernel(isa, mr, nr, unroll, prefetch)) {
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        
        std::cout << "   " << (found ? "✅" : "❌") 
                  << " tile" << mr << "×" << nr << " (" << desc << ")\n";
    }
    
    // ISA breakdown
    std::cout << "\n📊 ISA Breakdown:\n";
    int avx512_count = 0;
    int avx2_count = 0;
    
    // Sample a few configurations to count
    std::vector<std::string> isa_tags = {"simd::AVX512Tag", "simd::AVX2Tag"};
    std::vector<int> mr_vals = {1, 2, 4, 8, 16, 32, 64};
    std::vector<int> nr_vals = {1, 2, 4, 6, 8, 16, 32, 64};
    std::vector<int> unroll_vals = {1, 2, 4, 8, 16};
    std::vector<int> prefetch_vals = {0, 1, 2, 3, 5};
    
    for (const auto& isa : isa_tags) {
        for (int mr : mr_vals) {
            for (int nr : nr_vals) {
                for (int unroll : unroll_vals) {
                    for (int prefetch : prefetch_vals) {
                        if (registry.has_kernel(isa, mr, nr, unroll, prefetch)) {
                            if (isa == "simd::AVX512Tag") {
                                avx512_count++;
                            } else {
                                avx2_count++;
                            }
                        }
                    }
                }
            }
        }
    }
    
    std::cout << "   AVX512: " << avx512_count << " kernels\n";
    std::cout << "   AVX2:   " << avx2_count << " kernels\n";
    
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Verification " << (total == 1225 ? "PASSED ✅" : "FAILED ❌");
    std::cout << "                                    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    
    return (total == 1225) ? 0 : 1;
}
