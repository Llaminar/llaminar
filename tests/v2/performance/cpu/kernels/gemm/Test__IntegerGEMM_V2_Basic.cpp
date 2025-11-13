/**
 * @file Test__IntegerGEMM_V2_Basic.cpp
 * @brief Basic correctness test for V2 Integer GEMM rewrite
 *
 * Purpose:
 * - Validate V2 implementation against scalar reference
 * - Test small matrices first (debugging friendly)
 * - Verify VNNI path if available
 *
 * @author David Sanftenberg
 * @date November 11, 2025
 */

#include "kernels/cpu/gemm_v2/IntegerGemmKernelTemplateV2.h"
#include "kernels/cpu/gemm/GemmWeightCache.h"
#include "kernels/cpu/SimdTraits.h"
#include "tensors/Tensors.h"
#include <gtest/gtest.h>
#include <random>
#include <cmath>

using namespace llaminar2;
using namespace llaminar2::kernels::gemm;
using namespace llaminar2::kernels::simd;

// Simple Q8_0 block provider for testing
struct SimpleBlockProvider : public Q8_0BlockProvider
{
    const Q8_0Block *blocks;
    size_t k_blocks_;
    size_t num_rows_;

    SimpleBlockProvider(const Q8_0Block *b, size_t kb, size_t num_rows)
        : blocks(b), k_blocks_(kb), num_rows_(num_rows) {}

    const Q8_0Block *get_q8_block(size_t row_idx, size_t k_block_offset) override
    {
        return &blocks[row_idx * k_blocks_ + k_block_offset];
    }

    void warmup_cache(size_t, size_t, size_t, size_t) override {}
    size_t k_blocks() const override { return k_blocks_; }
    size_t num_rows() const override { return num_rows_; }
    bool is_zero_copy() const override { return true; }
};

// Helper: Generate random Q8_0 blocks
void generate_random_q8_blocks(Q8_0Block *blocks, size_t num_blocks, std::mt19937 &rng)
{
    std::uniform_int_distribution<int> dist(-127, 127);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);

    for (size_t i = 0; i < num_blocks; ++i)
    {
        blocks[i].d = fp32_to_fp16(scale_dist(rng));
        for (int j = 0; j < 32; ++j)
        {
            blocks[i].qs[j] = static_cast<int8_t>(dist(rng));
        }
    }
}

// Scalar reference implementation
void scalar_reference_gemm(
    const Q8_0Block *A, const Q8_0Block *B, Q8_0Block *C_ref,
    int m, int n, int k)
{
    const size_t k_blocks = (k + 31) / 32;
    const size_t n_blocks = (n + 31) / 32;

    std::vector<float> fp_acc(m * n_blocks * 32, 0.0f);

    for (int i = 0; i < m; ++i)
    {
        for (size_t nb = 0; nb < n_blocks; ++nb)
        {
            for (int j_in_block = 0; j_in_block < 32; ++j_in_block)
            {
                const int j = nb * 32 + j_in_block;
                if (j >= n)
                    continue;

                float sum = 0.0f;
                for (size_t kb = 0; kb < k_blocks; ++kb)
                {
                    const Q8_0Block &a_block = A[i * k_blocks + kb];
                    const Q8_0Block &b_block = B[j * k_blocks + kb];

                    float a_scale = fp16_to_fp32(a_block.d);
                    float b_scale = fp16_to_fp32(b_block.d);

                    int32_t dot = 0;
                    for (int kk = 0; kk < 32; ++kk)
                    {
                        dot += static_cast<int32_t>(a_block.qs[kk]) *
                               static_cast<int32_t>(b_block.qs[kk]);
                    }
                    sum += static_cast<float>(dot) * a_scale * b_scale;
                }
                fp_acc[i * n_blocks * 32 + nb * 32 + j_in_block] = sum;
            }
        }
    }

    // Quantize each row to Q8_0 blocks
    for (int i = 0; i < m; ++i)
    {
        for (size_t nb = 0; nb < n_blocks; ++nb)
        {
            const size_t row_offset = i * n_blocks * 32 + nb * 32;

            float amax = 0.0f;
            for (int jj = 0; jj < 32; ++jj)
            {
                float val = std::fabs(fp_acc[row_offset + jj]);
                if (val > amax)
                    amax = val;
            }

            float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
            float inv_scale = 1.0f / scale;
            C_ref[i * n_blocks + nb].d = fp32_to_fp16(scale);

            for (int jj = 0; jj < 32; ++jj)
            {
                float scaled = fp_acc[row_offset + jj] * inv_scale;
                int32_t q = static_cast<int32_t>(std::lroundf(scaled));
                q = std::max(-127, std::min(127, q));
                C_ref[i * n_blocks + nb].qs[jj] = static_cast<int8_t>(q);
            }
        }
    }
}

// Compare Q8_0 blocks
void compare_q8_blocks(
    const Q8_0Block *C_test, const Q8_0Block *C_ref,
    int m, int n,
    int &scale_mismatches, int &code_mismatches)
{
    const size_t n_blocks = (n + 31) / 32;
    scale_mismatches = 0;
    code_mismatches = 0;

    for (int i = 0; i < m; ++i)
    {
        for (size_t nb = 0; nb < n_blocks; ++nb)
        {
            const Q8_0Block &test = C_test[i * n_blocks + nb];
            const Q8_0Block &ref = C_ref[i * n_blocks + nb];

            float test_scale = fp16_to_fp32(test.d);
            float ref_scale = fp16_to_fp32(ref.d);

            if (std::fabs(test_scale - ref_scale) > 1e-5f)
            {
                scale_mismatches++;
            }

            for (int j = 0; j < 32; ++j)
            {
                if (test.qs[j] != ref.qs[j])
                {
                    code_mismatches++;
                }
            }
        }
    }
}

TEST(IntegerGEMMV2, TinyMatrix_1x32x32)
{
    std::mt19937 rng(42);

    const int m = 1, n = 32, k = 32;
    const size_t k_blocks = 1;
    const size_t n_blocks = 1;

    std::vector<Q8_0Block> A(m * k_blocks);
    std::vector<Q8_0Block> B(n * k_blocks);
    std::vector<Q8_0Block> C(m * n_blocks);
    std::vector<Q8_0Block> C_ref(m * n_blocks);

    generate_random_q8_blocks(A.data(), A.size(), rng);
    generate_random_q8_blocks(B.data(), B.size(), rng);

    SimpleBlockProvider B_provider(B.data(), k_blocks, n);

    // Run V2 kernel
    using KernelType = IntegerGemmKernelV2<AVX512VNNITag, 4, 32>;
    bool success = KernelType::multiply(A.data(), B_provider, C.data(), m, n, k);
    ASSERT_TRUE(success);

    // Run reference
    scalar_reference_gemm(A.data(), B.data(), C_ref.data(), m, n, k);

    // Compare
    int scale_mismatches = 0, code_mismatches = 0;
    compare_q8_blocks(C.data(), C_ref.data(), m, n, scale_mismatches, code_mismatches);

    EXPECT_EQ(scale_mismatches, 0) << "Scale mismatches detected";
    EXPECT_EQ(code_mismatches, 0) << "Code mismatches detected";
}

TEST(IntegerGEMMV2, SmallMatrix_4x32x64)
{
    std::mt19937 rng(43);

    const int m = 4, n = 32, k = 64;
    const size_t k_blocks = 2;
    const size_t n_blocks = 1;

    std::vector<Q8_0Block> A(m * k_blocks);
    std::vector<Q8_0Block> B(n * k_blocks);
    std::vector<Q8_0Block> C(m * n_blocks);
    std::vector<Q8_0Block> C_ref(m * n_blocks);

    generate_random_q8_blocks(A.data(), A.size(), rng);
    generate_random_q8_blocks(B.data(), B.size(), rng);

    SimpleBlockProvider B_provider(B.data(), k_blocks, n);

    using KernelType = IntegerGemmKernelV2<AVX512VNNITag, 4, 32>;
    bool success = KernelType::multiply(A.data(), B_provider, C.data(), m, n, k);
    ASSERT_TRUE(success);

    scalar_reference_gemm(A.data(), B.data(), C_ref.data(), m, n, k);

    int scale_mismatches = 0, code_mismatches = 0;
    compare_q8_blocks(C.data(), C_ref.data(), m, n, scale_mismatches, code_mismatches);

    EXPECT_EQ(scale_mismatches, 0) << "Scale mismatches detected";
    EXPECT_EQ(code_mismatches, 0) << "Code mismatches detected";
}

TEST(IntegerGEMMV2, MediumMatrix_16x64x128)
{
    std::mt19937 rng(44);

    const int m = 16, n = 64, k = 128;
    const size_t k_blocks = 4;
    const size_t n_blocks = 2;

    std::vector<Q8_0Block> A(m * k_blocks);
    std::vector<Q8_0Block> B(n * k_blocks);
    std::vector<Q8_0Block> C(m * n_blocks);
    std::vector<Q8_0Block> C_ref(m * n_blocks);

    generate_random_q8_blocks(A.data(), A.size(), rng);
    generate_random_q8_blocks(B.data(), B.size(), rng);

    SimpleBlockProvider B_provider(B.data(), k_blocks, n);

    using KernelType = IntegerGemmKernelV2<AVX512VNNITag, 16, 32>;
    bool success = KernelType::multiply(A.data(), B_provider, C.data(), m, n, k);
    ASSERT_TRUE(success);

    scalar_reference_gemm(A.data(), B.data(), C_ref.data(), m, n, k);

    int scale_mismatches = 0, code_mismatches = 0;
    compare_q8_blocks(C.data(), C_ref.data(), m, n, scale_mismatches, code_mismatches);

    EXPECT_EQ(scale_mismatches, 0) << "Scale mismatches detected";
    EXPECT_EQ(code_mismatches, 0) << "Code mismatches detected";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
