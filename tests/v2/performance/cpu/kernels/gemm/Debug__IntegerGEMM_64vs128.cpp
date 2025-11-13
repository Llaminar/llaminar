/**
 * @file Debug__IntegerGEMM_64vs128.cpp
 * @brief Debug test comparing 64-byte vs 128-byte mode results
 */

#include "kernels/cpu/gemm_v2/IntegerGemmKernelTemplateV2.h"
#include "tensors/Tensors.h"
#include <gtest/gtest.h>
#include <random>
#include <iostream>
#include <iomanip>

using namespace llaminar2;
using namespace llaminar2::kernels::gemm;
using namespace llaminar2::kernels::simd;

/**
 * @brief Simple Q8_0BlockProvider for testing - wraps a Q8_0Block array
 */
class SimpleBlockProvider : public Q8_0BlockProvider
{
private:
    const Q8_0Block *blocks_;
    size_t k_blocks_;
    size_t n_;

public:
    SimpleBlockProvider(const Q8_0Block *blocks, size_t k_blocks, size_t n)
        : blocks_(blocks), k_blocks_(k_blocks), n_(n) {}

    const Q8_0Block *get_q8_block(size_t row_idx, size_t k_block_offset) override
    {
        return &blocks_[row_idx * k_blocks_ + k_block_offset];
    }

    bool is_zero_copy() const override
    {
        return true; // Direct access to Q8_0 blocks
    }

    size_t k_blocks() const override
    {
        return k_blocks_;
    }

    size_t num_rows() const override
    {
        return n_;
    }
};

void generate_random_q8_blocks(Q8_0Block *blocks, size_t count, std::mt19937 &rng)
{
    std::uniform_int_distribution<int> dist(-127, 127);
    std::uniform_real_distribution<float> scale_dist(0.01f, 1.0f);

    for (size_t i = 0; i < count; ++i)
    {
        for (int j = 0; j < 32; ++j)
        {
            blocks[i].qs[j] = static_cast<int8_t>(dist(rng));
        }
        blocks[i].d = fp32_to_fp16(scale_dist(rng));
    }
}

void compare_blocks(const Q8_0Block *a, const Q8_0Block *b, size_t count, const char *name)
{
    int scale_mismatches = 0;
    int code_mismatches = 0;
    float max_scale_diff = 0.0f;

    for (size_t i = 0; i < count; ++i)
    {
        float scale_a = fp16_to_fp32(a[i].d);
        float scale_b = fp16_to_fp32(b[i].d);
        float scale_diff = std::abs(scale_a - scale_b);
        max_scale_diff = std::max(max_scale_diff, scale_diff);

        if (scale_diff > 1e-5f)
        {
            scale_mismatches++;
            if (scale_mismatches <= 3)
            {
                std::cout << "  Block " << i << " scale mismatch: "
                          << scale_a << " vs " << scale_b << " (diff=" << scale_diff << ")\n";
            }
        }

        for (int j = 0; j < 32; ++j)
        {
            if (a[i].qs[j] != b[i].qs[j])
            {
                code_mismatches++;
                if (code_mismatches <= 3)
                {
                    std::cout << "  Block " << i << "[" << j << "] code mismatch: "
                              << (int)a[i].qs[j] << " vs " << (int)b[i].qs[j] << "\n";
                }
            }
        }
    }

    std::cout << name << " comparison:\n";
    std::cout << "  Scale mismatches: " << scale_mismatches << " / " << count << "\n";
    std::cout << "  Code mismatches: " << code_mismatches << " / " << (count * 32) << "\n";
    std::cout << "  Max scale diff: " << max_scale_diff << "\n";
}

TEST(IntegerGEMM_64vs128, QprojPrefill32)
{
    std::mt19937 rng(42);

    // Q_proj (prefill-32): [32, 896] × [896, 896] → [32, 896]
    const int m = 32;
    const int k = 896;
    const int n = 896;

    const size_t k_blocks = (k + 31) / 32; // 28 blocks
    const size_t n_blocks = (n + 31) / 32; // 28 blocks

    std::cout << "\nTest: Q_proj (prefill-32) - " << m << "×" << k << "×" << n << "\n";
    std::cout << "K-blocks: " << k_blocks << " (28 blocks)\n";
    std::cout << "  With K_BLOCKS_PER_ITER=2: 14 iterations of 2 blocks\n";
    std::cout << "  With K_BLOCKS_PER_ITER=4: 7 iterations of 4 blocks\n\n";

    // Allocate matrices
    std::vector<Q8_0Block> A(m * k_blocks);
    std::vector<Q8_0Block> B(n * k_blocks);
    std::vector<Q8_0Block> C_64(m * n_blocks);
    std::vector<Q8_0Block> C_128(m * n_blocks);

    // Generate same random data for both
    generate_random_q8_blocks(A.data(), A.size(), rng);
    generate_random_q8_blocks(B.data(), B.size(), rng);

    SimpleBlockProvider B_provider(B.data(), k_blocks, n);

    // Test 64-byte kernel
    using Kernel64 = IntegerGemmKernelV2<AVX512VNNITag, 16, 32, 2>;
    bool success_64 = Kernel64::multiply(A.data(), B_provider, C_64.data(), m, n, k);
    ASSERT_TRUE(success_64) << "64-byte kernel failed";

    // Test 128-byte kernel
    using Kernel128 = IntegerGemmKernelV2<AVX512VNNITag, 16, 32, 4>;
    bool success_128 = Kernel128::multiply(A.data(), B_provider, C_128.data(), m, n, k);
    ASSERT_TRUE(success_128) << "128-byte kernel failed";

    // Compare results
    compare_blocks(C_64.data(), C_128.data(), m * n_blocks, "64-byte vs 128-byte");

    // They should be IDENTICAL
    for (size_t i = 0; i < C_64.size(); ++i)
    {
        EXPECT_EQ(C_64[i].d, C_128[i].d) << "Scale mismatch at block " << i;
        for (int j = 0; j < 32; ++j)
        {
            EXPECT_EQ(C_64[i].qs[j], C_128[i].qs[j])
                << "Code mismatch at block " << i << ", element " << j;
        }
    }
}

TEST(IntegerGEMM_64vs128, QprojPrefill128)
{
    std::mt19937 rng(43); // Different seed

    const int m = 128;
    const int k = 896;
    const int n = 896;

    const size_t k_blocks = (k + 31) / 32;
    const size_t n_blocks = (n + 31) / 32;

    std::cout << "\nTest: Q_proj (prefill-128) - " << m << "×" << k << "×" << n << "\n";

    std::vector<Q8_0Block> A(m * k_blocks);
    std::vector<Q8_0Block> B(n * k_blocks);
    std::vector<Q8_0Block> C_64(m * n_blocks);
    std::vector<Q8_0Block> C_128(m * n_blocks);

    generate_random_q8_blocks(A.data(), A.size(), rng);
    generate_random_q8_blocks(B.data(), B.size(), rng);

    SimpleBlockProvider B_provider(B.data(), k_blocks, n);

    using Kernel64 = IntegerGemmKernelV2<AVX512VNNITag, 16, 32, 2>;
    using Kernel128 = IntegerGemmKernelV2<AVX512VNNITag, 16, 32, 4>;

    ASSERT_TRUE(Kernel64::multiply(A.data(), B_provider, C_64.data(), m, n, k));
    ASSERT_TRUE(Kernel128::multiply(A.data(), B_provider, C_128.data(), m, n, k));

    compare_blocks(C_64.data(), C_128.data(), m * n_blocks, "64-byte vs 128-byte");

    for (size_t i = 0; i < C_64.size(); ++i)
    {
        EXPECT_EQ(C_64[i].d, C_128[i].d) << "Scale mismatch at block " << i;
        for (int j = 0; j < 32; ++j)
        {
            EXPECT_EQ(C_64[i].qs[j], C_128[i].qs[j])
                << "Code mismatch at block " << i << ", element " << j;
        }
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
