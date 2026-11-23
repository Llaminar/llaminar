/**
 * @file Test__Q8_1Tensor.cpp
 * @brief Unit tests for Q8_1Tensor quantized activation storage
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <array>
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"
#include <vector>
#include "../../utils/CPUProjectionTestWorkspace.h"
#include <random>
#include <cstring>
#include "v2/utils/MPIContext.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/kernels/cpu/gemm/FloatingPointGemmKernel.h"

using namespace llaminar2;

namespace
{
    /** @brief Construct actual mutable activation storage, with native per-row padding. */
    std::shared_ptr<Q8_1Tensor> create_empty_q8_1_tensor(int rows, int cols)
    {
        return std::make_shared<Q8_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});
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
 * Compares CPUNativeQuantisedGemmKernel (INT8) against FloatingPointGemmKernel (FP32 OneDNN)
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
    llaminar2::test::CPUProjectionTestWorkspace workspace(m, k, llaminar2::test::cpuProjectionTestRequirements(m, {quantized_gemm.get()}));
    ASSERT_TRUE(quantized_gemm->multiply_tensor(
        input.get(),
        output_quantized.get(),
        m, n, k, true, 1.f, 0.f, nullptr, nullptr, -1, workspace.get()));

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

    // The output is an activation buffer; immutable weight construction must
    // never be used to obtain writable block storage.
    auto q8_1_output = create_empty_q8_1_tensor(rows, cols);

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

/** @test FP32 producers publish all padded native blocks of each independent source row. */
TEST(Q8_1RowPublication, OddWidthWritesPerRowBlocks)
{
    using namespace llaminar2;
    constexpr size_t rows = 3, columns = 33;
    constexpr size_t blocks_per_row = 2;
    std::array<float, rows * columns> source{};
    for (size_t row = 0; row < rows; ++row)
        for (size_t column = 0; column < columns; ++column)
            source[row * columns + column] = static_cast<float>(3 + 19 * row) * (column < 20 ? 0.25f : -0.5f);

    std::array<Q8_1Block, rows * blocks_per_row> expected{};
    for (size_t row = 0; row < rows; ++row)
    {
        simd::quantize_single_block(source.data() + row * columns, expected[row * blocks_per_row], 32);
        simd::quantize_single_block(source.data() + row * columns + 32, expected[row * blocks_per_row + 1], 1);
    }
    Q8_1Tensor mutable_tensor({rows, columns}, DeviceId::cpu());
    ASSERT_TRUE(mutable_tensor.copyFrom_fp32_rows(source.data(), rows));
    ASSERT_EQ(mutable_tensor.size_bytes(), sizeof(expected));
    EXPECT_EQ(std::memcmp(mutable_tensor.raw_data(), expected.data(), sizeof(expected)), 0);

    Q8_1Tensor retained({rows + 1, columns}, DeviceId::cpu());
    auto *retained_bytes = static_cast<uint8_t *>(retained.raw_mutable_data());
    ASSERT_NE(retained_bytes, nullptr);
    std::array<uint8_t, blocks_per_row * sizeof(Q8_1Block)> guard;
    guard.fill(0x6b);
    std::memcpy(retained_bytes + sizeof(expected), guard.data(), guard.size());
    ASSERT_TRUE(retained.copyFrom_fp32_rows(source.data(), rows));
    const auto *published = static_cast<const uint8_t *>(retained.raw_data());
    EXPECT_EQ(std::memcmp(published, expected.data(), sizeof(expected)), 0);
    EXPECT_EQ(std::memcmp(published + sizeof(expected), guard.data(), guard.size()), 0);

    std::shared_ptr<Q8_1Tensor> immutable;
    ASSERT_NO_THROW(immutable = Q8_1Tensor::quantize_from_fp32(source.data(), {rows, columns}));
    ASSERT_TRUE(immutable);
    ASSERT_EQ(immutable->size_bytes(), sizeof(expected));
    EXPECT_EQ(std::memcmp(immutable->raw_data(), expected.data(), sizeof(expected)), 0);
}
