/**
 * @file Debug__IntegerGEMM_V2_VNNI.cpp
 * @brief Debug VNNI bias correction for V2 Integer GEMM
 */

#include "kernels/cpu/gemm_v2/IntegerGemmKernelTemplateV2.h"
#include "kernels/cpu/gemm/GemmWeightCache.h"
#include "kernels/cpu/SimdTraits.h"
#include "tensors/Tensors.h"
#include <iostream>
#include <iomanip>

using namespace llaminar2;
using namespace llaminar2::kernels::gemm;
using namespace llaminar2::kernels::simd;

// Simple Q8_0 block provider
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

int main()
{
    // Create simple 1x32x32 test
    const int m = 1, n = 32, k = 32;

    // Create simple test data
    Q8_0Block A_block, B_block;
    A_block.d = fp32_to_fp16(0.1f);
    B_block.d = fp32_to_fp16(0.1f);

    // Fill with simple pattern
    for (int i = 0; i < 32; ++i)
    {
        A_block.qs[i] = static_cast<int8_t>(i - 16); // [-16, 15]
        B_block.qs[i] = static_cast<int8_t>(i - 16); // [-16, 15]
    }

    std::vector<Q8_0Block> A(1);
    std::vector<Q8_0Block> B(32);
    std::vector<Q8_0Block> C(1);

    A[0] = A_block;
    for (int j = 0; j < 32; ++j)
    {
        B[j] = B_block;
    }

    // Compute scalar reference
    float a_scale = fp16_to_fp32(A_block.d);
    float b_scale = fp16_to_fp32(B_block.d);

    int32_t expected_dot = 0;
    for (int kk = 0; kk < 32; ++kk)
    {
        expected_dot += static_cast<int32_t>(A_block.qs[kk]) * static_cast<int32_t>(B_block.qs[kk]);
    }

    float expected_fp = static_cast<float>(expected_dot) * a_scale * b_scale;

    std::cout << "Expected INT32 dot product: " << expected_dot << "\n";
    std::cout << "Expected FP32 result: " << expected_fp << "\n";

    // Run V2 kernel
    SimpleBlockProvider B_provider(B.data(), 1, 32);
    using KernelType = IntegerGemmKernelV2<AVX512VNNITag, 4, 32>;
    bool success = KernelType::multiply(A.data(), B_provider, C.data(), m, n, k);

    if (!success)
    {
        std::cerr << "Kernel multiply failed!\n";
        return 1;
    }

    // Check result
    float result_scale = fp16_to_fp32(C[0].d);
    std::cout << "\nResult scale: " << result_scale << "\n";
    std::cout << "Result codes (first 10): ";
    for (int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<int>(C[0].qs[i]) << " ";
    }
    std::cout << "\n";

    // Dequantize and compare
    float result_fp = static_cast<float>(C[0].qs[0]) * result_scale;
    std::cout << "\nDequantized result[0]: " << result_fp << "\n";
    std::cout << "Expected: " << expected_fp << "\n";
    std::cout << "Error: " << std::abs(result_fp - expected_fp) << "\n";

    return 0;
}
