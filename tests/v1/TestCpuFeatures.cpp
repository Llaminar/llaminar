// Test CPU feature detection
#include <gtest/gtest.h>
#include "utils/CpuFeatures.h"
#include <iostream>

using namespace llaminar;

TEST(CpuFeaturesTest, DetectionWorks)
{
    const auto &features = CpuFeatures::instance();

    // Just print what we have - this is informational
    std::cout << "\n"
              << features.summary() << std::endl;

    // On any x86_64 system we should at least detect something
    // (even if it's just "no advanced SIMD")
    EXPECT_FALSE(features.summary().empty());

    // Log individual features
    std::cout << "  AVX512F: " << (features.has_avx512f() ? "YES" : "NO") << std::endl;
    std::cout << "  AVX512_BF16: " << (features.has_avx512_bf16() ? "YES" : "NO") << std::endl;
    std::cout << "  AVX512_FP16: " << (features.has_avx512_fp16() ? "YES" : "NO") << std::endl;
    std::cout << "  AVX512_VNNI: " << (features.has_avx512_vnni() ? "YES" : "NO") << std::endl;
    std::cout << "  F16C: " << (features.has_f16c() ? "YES" : "NO") << std::endl;

    // Test convenience function
    bool can_use_bf16 = can_use_native_bf16_gemm();
    std::cout << "\nCan use native BF16 GEMM: " << (can_use_bf16 ? "YES" : "NO") << std::endl;

    // Consistency check: if we have BF16, we should have AVX512F base
    if (features.has_avx512_bf16())
    {
        EXPECT_TRUE(features.has_avx512f()) << "AVX512_BF16 requires AVX512F base";
    }

    // Consistency check: convenience function should match direct check
    EXPECT_EQ(can_use_bf16, features.has_avx512_bf16());
}

TEST(CpuFeaturesTest, SingletonConsistency)
{
    // Multiple calls should return same instance with same results
    const auto &features1 = CpuFeatures::instance();
    const auto &features2 = CpuFeatures::instance();

    EXPECT_EQ(&features1, &features2) << "Should return same singleton instance";
    EXPECT_EQ(features1.has_avx512_bf16(), features2.has_avx512_bf16());
    EXPECT_EQ(features1.summary(), features2.summary());
}
