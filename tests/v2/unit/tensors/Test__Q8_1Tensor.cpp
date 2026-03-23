/**
 * @file Test__Q8_1Tensor.cpp
 * @brief Unit tests for Q8_1Tensor quantized activation storage
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"
#include <vector>
#include <random>
#include <cstring>
#include "v2/utils/MPIContext.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/kernels/cpu/gemm/FloatingPointGemmKernel.h"

using namespace llaminar2;

namespace
{
    std::shared_ptr<Q8_1Tensor> create_empty_q8_1_tensor(int rows, int cols)
    {
        const size_t total_elements = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        const size_t total_blocks = (total_elements + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        std::vector<uint8_t> raw(total_blocks * sizeof(Q8_1Block), 0);
        return std::make_shared<Q8_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
            raw);
    }
}

/**
 * @brief Ensure Q8_1Tensor::from_int32_with_scales scales/quantizes activations and preserves precomputed sums.
 */
TEST(Test__Q8_1Tensor, FromInt32WithScalesQuantizesAndStoresSum)
{
    constexpr int rows = 1;
    constexpr int cols = 4;
    auto tensor = create_empty_q8_1_tensor(rows, cols);

    const std::vector<int32_t> accum = {8, -8, 16, -16};
    const std::vector<float> row_scales = {0.5f};
    const std::vector<float> col_scales = {1.0f, 1.0f, 0.5f, 0.5f};
    const std::vector<float> bias = {0.0f, 0.5f, 0.0f, -0.5f};

    ASSERT_TRUE(tensor->from_int32_with_scales(
        accum.data(),
        rows,
        cols,
        row_scales.data(),
        col_scales.data(),
        bias.data()));

    std::vector<float> dequant(rows * cols, 0.0f);
    tensor->to_fp32(dequant.data());

    const std::vector<float> expected = {
        4.0f,  // 8 * 0.5 * 1.0 + 0.0
        -3.5f, // -8 * 0.5 * 1.0 + 0.5
        4.0f,  // 16 * 0.5 * 0.5 + 0.0
        -4.5f  // -16 * 0.5 * 0.5 - 0.5
    };

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(dequant[i], expected[i], 0.2f) << "Mismatch at index " << i;
    }

    const Q8_1Block *block = tensor->decode_to_q8_1(0, 0);
    ASSERT_NE(block, nullptr);

    int32_t raw_sum = 0;
    for (size_t i = 0; i < Q8_1Block::BLOCK_SIZE; ++i)
    {
        raw_sum += block->qs[i];
    }
    EXPECT_EQ(block->sum_qs, raw_sum);
    EXPECT_GT(fp16_to_fp32(block->d), 0.0f);
    if (cols < static_cast<int>(Q8_1Block::BLOCK_SIZE))
    {
        EXPECT_EQ(block->qs[cols], 0) << "First padded element should remain zero";
    }
}

/**
 * @brief Quantized GEMM vs FP32 GEMM Parity Test for Q8_1
 *
 * Compares CPUNativeVNNIGemmKernel (INT8) against FloatingPointGemmKernel (FP32 OneDNN)
 * using randomly initialized Q8_1 weights. Validates that quantization introduces
 * acceptable error (< 1% relative L2).
 */
TEST(Test__Q8_1Tensor, QuantizedVsFP32Parity)
{
    using namespace llaminar2;

    // Create MPI context for tensor factory
    auto mpi_ctx = std::make_unique<MPIContext>(0, 1, MPI_COMM_WORLD);

    // Realistic dimensions: 64 tokens, 512 hidden dim
    const int m = 64;
    const int n = 512;
    const int k = 512;

    // Q8_1: 32 elements per block (scale + sum_qs + 32 int8 values)
    const size_t num_blocks = (static_cast<size_t>(n) * k) / 32;
    std::vector<uint8_t> raw_data(num_blocks * sizeof(Q8_1Block));
    Q8_1Block *blocks = reinterpret_cast<Q8_1Block *>(raw_data.data());

    // Initialize with random but valid data
    std::mt19937 rng(42);
    std::uniform_int_distribution<int8_t> int8_dist(-127, 127);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);

    for (size_t b = 0; b < num_blocks; ++b)
    {
        // Valid FP16 scale factor
        blocks[b].d = fp32_to_fp16(scale_dist(rng));
        // Random int8 values and compute sum
        int32_t sum = 0;
        for (int i = 0; i < 32; ++i)
        {
            blocks[b].qs[i] = int8_dist(rng);
            sum += blocks[b].qs[i];
        }
        blocks[b].sum_qs = static_cast<int16_t>(sum);
    }

    // Create quantized tensor
    auto q8_1_tensor = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)},
        raw_data);

    // Dequantize to FP32 for reference
    TensorFactory factory(*mpi_ctx);
    auto fp32_weights = factory.createFP32({static_cast<size_t>(n), static_cast<size_t>(k)});
    q8_1_tensor->to_fp32(fp32_weights->mutable_data());

    // Create random input activations
    auto input = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(k)});
    float *input_data = input->mutable_data();
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    for (int i = 0; i < m * k; ++i)
    {
        input_data[i] = input_dist(rng);
    }

    // Allocate outputs
    auto output_quantized = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
    auto output_fp32 = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
    std::memset(output_quantized->mutable_data(), 0, m * n * sizeof(float));
    std::memset(output_fp32->mutable_data(), 0, m * n * sizeof(float));

    // Run quantized GEMM (INT8 path)
    auto quantized_gemm = q8_1_tensor->createGemm();
    ASSERT_TRUE(quantized_gemm->multiply_tensor(
        input.get(),
        output_quantized.get(),
        m, n, k));

    // Run FP32 GEMM (OneDNN reference)
    gemm::FloatingPointGemmKernel fp32_gemm(fp32_weights.get());
    ASSERT_TRUE(fp32_gemm.multiply_tensor(
        input.get(),
        output_fp32.get(),
        m, n, k));

    // Compare results - compute relative L2 error
    double sum_sq_diff = 0.0, sum_sq_ref = 0.0;
    for (int i = 0; i < m * n; ++i)
    {
        double diff = static_cast<double>(output_quantized->data()[i]) - static_cast<double>(output_fp32->data()[i]);
        sum_sq_diff += diff * diff;
        sum_sq_ref += static_cast<double>(output_fp32->data()[i]) * static_cast<double>(output_fp32->data()[i]);
    }
    float rel_l2_error = (sum_sq_ref > 0) ? static_cast<float>(std::sqrt(sum_sq_diff / sum_sq_ref)) : 0.0f;

    std::cout << "[Q8_1 Parity] Relative L2 error: " << (rel_l2_error * 100.0f) << "%" << std::endl;

    // 1% tolerance for quantization error
    EXPECT_LT(rel_l2_error, 0.01f)
        << "Q8_1 quantized GEMM error exceeds 1% threshold";
}

/**
 * @brief Test q8_1_add_q8_1 SIMD primitive for native Q8_1 residual addition
 *
 * Tests the Q8_1 + Q8_1 -> Q8_1 operation used by TypedResidualOp:
 * 1. Create two Q8_1 tensors from FP32 data
 * 2. Add using q8_1_add_q8_1 SIMD primitive
 * 3. Dequantize result and compare to FP32 reference
 *
 * Expected error: Slightly higher than single-tensor quantization due to
 * two quantization operations (A and B) plus one requantization (output).
 */
TEST(Test__Q8_1Tensor, NativeQ8_1Addition)
{
    using namespace llaminar2;

    const int rows = 4;
    const int cols = 64; // Must be multiple of 32 (Q8_1 block size)
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random FP32 data for A and B
    std::vector<float> fp32_a(elements);
    std::vector<float> fp32_b(elements);
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        fp32_a[i] = dist(gen);
        fp32_b[i] = dist(gen);
    }

    // Compute FP32 reference: expected = A + B
    std::vector<float> expected(elements);
    for (size_t i = 0; i < elements; ++i)
    {
        expected[i] = fp32_a[i] + fp32_b[i];
    }

    // Quantize A and B to Q8_1
    auto q8_1_a = Q8_1Tensor::quantize_from_fp32(
        fp32_a.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});
    auto q8_1_b = Q8_1Tensor::quantize_from_fp32(
        fp32_b.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(q8_1_a, nullptr);
    ASSERT_NE(q8_1_b, nullptr);

    // Create output tensor
    const size_t n_blocks = elements / 32;
    std::vector<uint8_t> raw_output(n_blocks * sizeof(Q8_1Block), 0);
    auto q8_1_output = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        raw_output);

    // Use q8_1_add_q8_1 SIMD primitive
    simd::q8_1_add_q8_1(
        q8_1_a->q8_1_blocks(),
        q8_1_b->q8_1_blocks(),
        q8_1_output->mutable_q8_1_blocks(),
        elements);

    // Dequantize output and compare to FP32 reference
    std::vector<float> actual(elements);
    q8_1_output->to_fp32(actual.data());

    // Compute relative L2 error
    double sum_sq_diff = 0.0, sum_sq_ref = 0.0;
    for (size_t i = 0; i < elements; ++i)
    {
        double diff = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
        sum_sq_diff += diff * diff;
        sum_sq_ref += static_cast<double>(expected[i]) * static_cast<double>(expected[i]);
    }
    float rel_l2_error = (sum_sq_ref > 0) ? static_cast<float>(std::sqrt(sum_sq_diff / sum_sq_ref)) : 0.0f;

    std::cout << "[Q8_1 Addition] Relative L2 error: " << (rel_l2_error * 100.0f) << "%" << std::endl;

    // 3% tolerance (accounts for 3 quantization ops: A quant + B quant + output requant)
    EXPECT_LT(rel_l2_error, 0.03f)
        << "Q8_1 native addition error exceeds 3% threshold";

    // Also verify per-element error is bounded
    float max_abs_error = 0.0f;
    for (size_t i = 0; i < elements; ++i)
    {
        float abs_err = std::abs(actual[i] - expected[i]);
        max_abs_error = std::max(max_abs_error, abs_err);
    }
    std::cout << "[Q8_1 Addition] Max abs error: " << max_abs_error << std::endl;

    // Max error should be < 0.1 for values in [-2, 2] range
    EXPECT_LT(max_abs_error, 0.15f)
        << "Q8_1 native addition max error exceeds 0.15 threshold";
}
