/**
 * @file Test__GemmTemplateInfrastructure.cpp
 * @brief Phase 1 validation: Ensure template infrastructure compiles and works
 *
 * Tests:
 * - SimdTraits instantiation for AVX512, AVX2, Scalar
 * - MicroKernel instantiation with various tile sizes
 * - GemmKernel instantiation with various parameters
 * - Basic operations (zero, accumulate, reduce)
 *
 * @author David Sanftenberg
 * @date October 2025
 */

#include <gtest/gtest.h>
#include "../../src/v2/kernels/cpu/gemm/GemmMicroKernel.h"
#include "../../src/v2/kernels/cpu/SimdTraits.h"
#include "../../src/v2/kernels/cpu/gemm/GemmKernelTemplate.h"
#include <cmath>
#include <vector>

using namespace llaminar2::kernels;

// ========== SIMD TRAITS TESTS ==========

TEST(GemmTemplateInfrastructure, SimdTraits_Scalar)
{
    using Traits = simd::SimdTraits<simd::ScalarTag>;

    EXPECT_EQ(Traits::vector_width, 1);
    EXPECT_STREQ(Traits::isa_name, "Scalar");

    auto zero = Traits::zero();
    EXPECT_EQ(Traits::reduce_add(zero), 0.0f);

    simd::ScalarVector a(2.0f);
    simd::ScalarVector b(3.0f);
    simd::ScalarVector c(1.0f);

    auto result = Traits::fmadd(a, b, c); // 2*3 + 1 = 7
    EXPECT_EQ(Traits::reduce_add(result), 7.0f);
}

#if defined(__AVX512F__)
TEST(GemmTemplateInfrastructure, SimdTraits_AVX512)
{
    using Traits = simd::SimdTraits<simd::AVX512Tag>;

    EXPECT_EQ(Traits::vector_width, 16);
    EXPECT_STREQ(Traits::isa_name, "AVX512");

    auto zero = Traits::zero();
    EXPECT_EQ(Traits::reduce_add(zero), 0.0f);

    alignas(64) float a_data[16], b_data[16], c_data[16];
    for (int i = 0; i < 16; ++i)
    {
        a_data[i] = 2.0f;
        b_data[i] = 3.0f;
        c_data[i] = 1.0f;
    }

    auto a = Traits::load(a_data);
    auto b = Traits::load(b_data);
    auto c = Traits::load(c_data);

    auto result = Traits::fmadd(a, b, c); // 2*3 + 1 = 7 per element
    float sum = Traits::reduce_add(result);
    EXPECT_FLOAT_EQ(sum, 7.0f * 16); // 16 elements × 7 = 112
}
#endif

#if defined(__AVX2__)
TEST(GemmTemplateInfrastructure, SimdTraits_AVX2)
{
    using Traits = simd::SimdTraits<simd::AVX2Tag>;

    EXPECT_EQ(Traits::vector_width, 8);
    EXPECT_STREQ(Traits::isa_name, "AVX2");

    auto zero = Traits::zero();
    EXPECT_EQ(Traits::reduce_add(zero), 0.0f);

    alignas(32) float a_data[8], b_data[8], c_data[8];
    for (int i = 0; i < 8; ++i)
    {
        a_data[i] = 2.0f;
        b_data[i] = 3.0f;
        c_data[i] = 1.0f;
    }

    auto a = Traits::load(a_data);
    auto b = Traits::load(b_data);
    auto c = Traits::load(c_data);

    auto result = Traits::fmadd(a, b, c); // 2*3 + 1 = 7 per element
    float sum = Traits::reduce_add(result);
    EXPECT_FLOAT_EQ(sum, 7.0f * 8); // 8 elements × 7 = 56
}
#endif

// ========== MICRO-KERNEL TESTS ==========

TEST(GemmTemplateInfrastructure, MicroKernel_Scalar_4x4)
{
    using MK = gemm::MicroKernel<simd::ScalarTag, 4, 4>;

    MK ukernel;
    ukernel.zero();

    // Verify all accumulators are zero
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            EXPECT_EQ(simd::SimdTraits<simd::ScalarTag>::reduce_add(ukernel.accumulator(i, j)), 0.0f);
        }
    }

    // Simple accumulation test
    alignas(64) float A_panel[4 * 32]; // 4 rows × 32 cols
    alignas(64) float B_panel[4 * 32]; // 4 cols × 32 rows

    for (int i = 0; i < 4 * 32; ++i)
    {
        A_panel[i] = 1.0f;
        B_panel[i] = 1.0f;
    }

    // Accumulate 32 times (full K dimension)
    for (int p = 0; p < 32; p += 1)
    { // vector_width = 1 for scalar
        ukernel.accumulate(A_panel, B_panel, 32, p);
    }

    // Reduce to output
    float C_tile[16];
    ukernel.reduce(C_tile);

    // Each element should be 1*1*32 = 32
    for (int i = 0; i < 16; ++i)
    {
        EXPECT_FLOAT_EQ(C_tile[i], 32.0f);
    }
}

#if defined(__AVX512F__)
TEST(GemmTemplateInfrastructure, MicroKernel_AVX512_8x4)
{
    using MK = gemm::MicroKernel<simd::AVX512Tag, 8, 4>;

    MK ukernel;
    ukernel.zero();

    // Test with block_size=32 (2 iterations of 16-wide vectors)
    alignas(64) float A_panel[8 * 32];
    alignas(64) float B_panel[4 * 32];

    for (int i = 0; i < 8 * 32; ++i)
    {
        A_panel[i] = 2.0f;
    }
    for (int i = 0; i < 4 * 32; ++i)
    {
        B_panel[i] = 3.0f;
    }

    // Two iterations (32 elements / 16 per vector)
    for (int p = 0; p < 32; p += 16)
    {
        ukernel.accumulate(A_panel, B_panel, 32, p);
    }

    float C_tile[32];
    ukernel.reduce(C_tile);

    // Each element: 2*3 = 6 per FMA, 32 FMAs total (16+16)
    // Result: 6 * 32 = 192
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_FLOAT_EQ(C_tile[i], 192.0f);
    }
}

TEST(GemmTemplateInfrastructure, MicroKernel_AVX512_8x8)
{
    // Test NEW tile size (8×8 instead of 8×4)
    using MK = gemm::MicroKernel<simd::AVX512Tag, 8, 8>;

    MK ukernel;
    ukernel.zero();

    alignas(64) float A_panel[8 * 32];
    alignas(64) float B_panel[8 * 32]; // 8 columns now!

    for (int i = 0; i < 8 * 32; ++i)
    {
        A_panel[i] = 1.0f;
        B_panel[i] = 1.0f;
    }

    for (int p = 0; p < 32; p += 16)
    {
        ukernel.accumulate(A_panel, B_panel, 32, p);
    }

    float C_tile[64]; // 8×8 = 64 elements
    ukernel.reduce(C_tile);

    // Each element: 1*1*32 = 32
    for (int i = 0; i < 64; ++i)
    {
        EXPECT_FLOAT_EQ(C_tile[i], 32.0f);
    }
}
#endif

// ========== COMPILATION TESTS (just ensure these instantiate) ==========

TEST(GemmTemplateInfrastructure, GemmKernel_Instantiation)
{
    // These should all compile without errors

#if defined(__AVX512F__)
    using K1 = gemm::GemmKernel<simd::AVX512Tag, 8, 4, 8, 5>;
    using K2 = gemm::GemmKernel<simd::AVX512Tag, 8, 8, 8, 5>;  // NEW: 8×8
    using K3 = gemm::GemmKernel<simd::AVX512Tag, 8, 16, 8, 5>; // NEW: 8×16
    using K4 = gemm::GemmKernel<simd::AVX512Tag, 16, 8, 16, 5>;
    using K5 = gemm::GemmKernel<simd::AVX512Tag, 32, 16, 16, 5>;
    using K6 = gemm::GemmKernel<simd::AVX512Tag, 64, 32, 16, 5>; // Large tile
#endif

#if defined(__AVX2__)
    using K7 = gemm::GemmKernel<simd::AVX2Tag, 4, 4, 8, 5>;
    using K8 = gemm::GemmKernel<simd::AVX2Tag, 8, 4, 8, 5>;
    using K9 = gemm::GemmKernel<simd::AVX2Tag, 8, 8, 8, 5>; // NEW: 8×8
#endif

    using K10 = gemm::GemmKernel<simd::ScalarTag, 4, 4, 4, 3>;

    // If we get here, all templates instantiated successfully
    SUCCEED();
}

// ========== PREFERRED TILE SIZES ==========

#if defined(__AVX512F__)
TEST(GemmTemplateInfrastructure, PreferredTileSizes_AVX512)
{
    using Pref = gemm::PreferredTileSizes<simd::AVX512Tag>;

    EXPECT_EQ(Pref::small_m, 8);
    EXPECT_EQ(Pref::small_n, 4);
    EXPECT_EQ(Pref::medium_m, 16);
    EXPECT_EQ(Pref::medium_n, 8);
    EXPECT_EQ(Pref::large_m, 32);
    EXPECT_EQ(Pref::large_n, 16);
    EXPECT_EQ(Pref::xlarge_m, 64);
    EXPECT_EQ(Pref::xlarge_n, 32);
}
#endif

#if defined(__AVX2__)
TEST(GemmTemplateInfrastructure, PreferredTileSizes_AVX2)
{
    using Pref = gemm::PreferredTileSizes<simd::AVX2Tag>;

    EXPECT_EQ(Pref::small_m, 4);
    EXPECT_EQ(Pref::small_n, 4);
    EXPECT_EQ(Pref::medium_m, 8);
    EXPECT_EQ(Pref::medium_n, 4);
    EXPECT_EQ(Pref::large_m, 16);
    EXPECT_EQ(Pref::large_n, 8);
}
#endif
