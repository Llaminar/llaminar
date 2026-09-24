#include <gtest/gtest.h>
#include "v2/tensors/SIMDHelpers.h"
#include "v2/tensors/BlockStructures.h"
#include <vector>
#include <random>
#include <iostream>

using namespace llaminar2;
using namespace llaminar2::simd;

TEST(Test__Q3K_Bug, VerifyUnpackCorrectness)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX512 not supported";
    }

    // Create a Q3_K block
    Q3_KBlock block;
    // Initialize with deterministic values
    block.d = fp32_to_fp16(1.0f);

    // Set scales to random values to ensure sub-blocks have different parameters
    for (int i = 0; i < 12; ++i)
        block.scales[i] = i * 17;
    // Force SB0 and SB1 to be very different
    block.scales[1] = 0;   // Used by SB0, 2nd half
    block.scales[3] = 255; // Used by SB1, 2nd half

    for (int i = 0; i < 64; ++i)
        block.qs[i] = i * 3;
    for (int i = 0; i < 32; ++i)
        block.hmask[i] = i * 7;

    // Run AVX512
    int8_t output_avx[256];
    float scales_avx[8];
    float mins_avx[8];
    unpack_q3_k_superblock_to_int8_avx512(block, output_avx, scales_avx, mins_avx);

    // Run Scalar (reference)
    int8_t output_ref[256];
    float scales_ref[8];
    float mins_ref[8];

    // We need to call the scalar version.
    // unpack_q3_k_superblock_to_int8 calls the best available.
    // We can manually call transcode_q3_k_to_int8_scalar for each sub-block.
    for (int i = 0; i < 8; ++i)
    {
        transcode_q3_k_to_int8_scalar(block, i, output_ref + i * 32, &scales_ref[i], &mins_ref[i]);
    }

    // Compare
    for (int i = 0; i < 256; ++i)
    {
        EXPECT_EQ((int)output_avx[i], (int)output_ref[i]) << "Mismatch at index " << i;
    }
#else
    GTEST_SKIP() << "AVX512 unpacker not compiled";
#endif
}
