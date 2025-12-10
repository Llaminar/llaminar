#include <gtest/gtest.h>
#include "v2/tensors/SIMDHelpers.h"
#include "v2/tensors/Tensors.h"
#include <vector>
#include <cmath>
#include <random>

using namespace llaminar2;

class Test__Q8_1_Add_Repro : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }
};

TEST_F(Test__Q8_1_Add_Repro, LargeValues)
{
    Q8_1Block a, b, out;

    // Case 1: Large values (near saturation)
    float val_a = 1000.0f;
    float val_b = 2000.0f;

    // Quantize A
    simd::quantize_single_block_scalar(&val_a, a, 1); // Only 1 valid element, rest 0
    // Quantize B
    simd::quantize_single_block_scalar(&val_b, b, 1);

    // Add
    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    // Dequantize output
    float out_val = simd::fp16_to_fp32(out.d) * out.qs[0];

    EXPECT_NEAR(out_val, 3000.0f, 3000.0f * 0.05f); // 5% tolerance
}

TEST_F(Test__Q8_1_Add_Repro, SmallValues)
{
    Q8_1Block a, b, out;

    // Case 2: Small values
    float val_a = 1e-4f;
    float val_b = 2e-4f;

    simd::quantize_single_block_scalar(&val_a, a, 1);
    simd::quantize_single_block_scalar(&val_b, b, 1);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    float out_val = simd::fp16_to_fp32(out.d) * out.qs[0];

    EXPECT_NEAR(out_val, 3e-4f, 3e-4f * 0.1f); // 10% tolerance (quantization noise is higher for small values)
}

TEST_F(Test__Q8_1_Add_Repro, MixedScales)
{
    Q8_1Block a, b, out;

    // Case 3: Mixed scales (one large, one small)
    float val_a = 1000.0f;
    float val_b = 0.1f;

    simd::quantize_single_block_scalar(&val_a, a, 1);
    simd::quantize_single_block_scalar(&val_b, b, 1);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    float out_val = simd::fp16_to_fp32(out.d) * out.qs[0];

    EXPECT_NEAR(out_val, 1000.1f, 1000.1f * 0.01f);
}

TEST_F(Test__Q8_1_Add_Repro, Cancellation)
{
    Q8_1Block a, b, out;

    // Case 4: Cancellation (positive + negative)
    float val_a = 100.0f;
    float val_b = -99.0f;

    simd::quantize_single_block_scalar(&val_a, a, 1);
    simd::quantize_single_block_scalar(&val_b, b, 1);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    float out_val = simd::fp16_to_fp32(out.d) * out.qs[0];

    EXPECT_NEAR(out_val, 1.0f, 0.5f); // Absolute tolerance
}

TEST_F(Test__Q8_1_Add_Repro, FullBlockRandom)
{
    Q8_1Block a, b, out;
    float fa[32], fb[32];

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int i = 0; i < 32; ++i)
    {
        fa[i] = dist(gen);
        fb[i] = dist(gen);
    }

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    float d = simd::fp16_to_fp32(out.d);
    for (int i = 0; i < 32; ++i)
    {
        float val = d * out.qs[i];
        float expected = fa[i] + fb[i];
        // Quantization error can be up to scale/2.
        // Scale is max_abs/127.
        // Max abs of sum is approx 20. Scale approx 0.15. Error approx 0.075.
        EXPECT_NEAR(val, expected, 0.5f);
    }
}
