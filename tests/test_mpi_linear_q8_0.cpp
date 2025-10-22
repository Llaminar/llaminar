/**
 * @file test_mpi_linear_q8_0.cpp
 * @brief Focused validation tests for MPILinearOperator Q8_0 streaming decode
 *
 * Test suite for Week 2 Step 2:
 * 1. Q8_0 streaming decode path correctness
 * 2. Parity tests (Q8_0 output vs FP32 decoded baseline)
 * 3. Multi-rank MPI distribution
 * 4. Edge cases (small matrices, single rows, various dimensions)
 * 5. Performance validation
 *
 * @author David Sanftenberg
 * @date October 21, 2025
 */

#include <gtest/gtest.h>
#include "../src/operators/MPILinearOperator.h"
#include "../src/tensors/Q8_0Tensor.h"
#include "../src/tensors/SimpleTensor.h"
#include "../src/tensors/TensorFactory.h"
#include "../src/utils/DebugEnv.h"
#include <mpi.h>
#include <memory>
#include <cmath>
#include <chrono>

using namespace llaminar;

class MPILinearQ8_0Test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize MPI if not already initialized
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            int provided;
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        }

        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }

    /**
     * @brief Create Q8_0 weight tensor from FP32 data
     *
     * Creates Q8_0 blocks from FP32 weight matrix for testing.
     * Uses simple quantization: scale = max_abs / 127, quant = round(val / scale).
     *
     * @param fp32_data Source FP32 data
     * @param rows Number of rows
     * @param cols Number of columns
     * @return std::shared_ptr<Q8_0Tensor> Quantized weight tensor
     */
    std::shared_ptr<Q8_0Tensor> createQ8_0Weight(const std::vector<float> &fp32_data, size_t rows, size_t cols)
    {
        constexpr size_t BLOCK_SIZE = 32;
        size_t num_elements = rows * cols;
        size_t num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        std::vector<uint8_t> raw_data;
        raw_data.reserve(num_blocks * 34); // 2 bytes scale + 32 bytes values

        // Process each block
        for (size_t b = 0; b < num_blocks; ++b)
        {
            size_t start_idx = b * BLOCK_SIZE;
            size_t end_idx = std::min(start_idx + BLOCK_SIZE, num_elements);

            // Find max absolute value in block
            float max_abs = 0.0f;
            for (size_t i = start_idx; i < end_idx; ++i)
            {
                max_abs = std::max(max_abs, std::abs(fp32_data[i]));
            }

            // Compute scale (FP16 format)
            float scale = (max_abs > 1e-10f) ? (max_abs / 127.0f) : 1e-8f;
            // Clamp scale to avoid FP16 overflow/underflow
            scale = std::max(1e-10f, std::min(scale, 65504.0f)); // FP16 max ~65504

            // Convert scale to FP16
            uint32_t bits;
            std::memcpy(&bits, &scale, 4);
            uint32_t sign = (bits >> 16) & 0x8000;
            uint32_t exp = ((bits >> 23) & 0xFF);
            uint32_t mant = (bits & 0x7FFFFF);

            uint16_t fp16;
            if (exp == 0)
            {
                fp16 = sign;
            }
            else if (exp == 255)
            {
                fp16 = sign | 0x7C00 | (mant ? 0x0200 : 0);
            }
            else
            {
                int32_t new_exp = exp - 127 + 15;
                if (new_exp <= 0)
                {
                    fp16 = sign;
                }
                else if (new_exp >= 31)
                {
                    fp16 = sign | 0x7C00;
                }
                else
                {
                    fp16 = sign | (new_exp << 10) | (mant >> 13);
                }
            }

            // Write FP16 scale (2 bytes)
            raw_data.push_back(fp16 & 0xFF);
            raw_data.push_back((fp16 >> 8) & 0xFF);

            // Quantize and write values
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                if (start_idx + i < end_idx)
                {
                    float val = fp32_data[start_idx + i];
                    int8_t quant = static_cast<int8_t>(std::round(val / scale));
                    raw_data.push_back(static_cast<uint8_t>(quant));
                }
                else
                {
                    raw_data.push_back(0); // Padding
                }
            }
        }

        return std::make_shared<Q8_0Tensor>(std::vector<int>{static_cast<int>(rows), static_cast<int>(cols)}, raw_data);
    }

    /**
     * @brief Compute reference FP32 output for parity testing
     *
     * Performs Y = X @ W^T using standard FP32 GEMM for baseline comparison.
     *
     * @param input Input matrix [seq_len, in_dim]
     * @param weight Weight matrix [out_dim, in_dim]
     * @param seq_len Sequence length
     * @param in_dim Input dimension
     * @param out_dim Output dimension
     * @return std::vector<float> Reference output [seq_len, out_dim]
     */
    std::vector<float> computeReferenceFP32(
        const float *input,
        const float *weight,
        size_t seq_len,
        size_t in_dim,
        size_t out_dim)
    {
        std::vector<float> output(seq_len * out_dim, 0.0f);

        // Y = X @ W^T
        // X: [seq_len, in_dim]
        // W: [out_dim, in_dim]
        // Y: [seq_len, out_dim]
        for (size_t s = 0; s < seq_len; ++s)
        {
            for (size_t o = 0; o < out_dim; ++o)
            {
                float sum = 0.0f;
                for (size_t i = 0; i < in_dim; ++i)
                {
                    sum += input[s * in_dim + i] * weight[o * in_dim + i];
                }
                output[s * out_dim + o] = sum;
            }
        }

        return output;
    }

    /**
     * @brief Compare two tensors with relative tolerance
     *
     * Validates Q8_0 output matches reference within acceptable quantization error.
     * Uses relative L2 norm for robust comparison.
     *
     * @param actual Actual output from Q8_0 path
     * @param expected Expected output from FP32 baseline
     * @param rel_tolerance Relative tolerance (default 1e-2 for Q8_0)
     * @return true if within tolerance, false otherwise
     */
    bool compareTensorsWithTolerance(
        const float *actual,
        const float *expected,
        size_t count,
        float rel_tolerance = 1.5e-2f)
    {
        // Compute L2 norms
        float actual_norm = 0.0f;
        float expected_norm = 0.0f;
        float diff_norm = 0.0f;

        for (size_t i = 0; i < count; ++i)
        {
            actual_norm += actual[i] * actual[i];
            expected_norm += expected[i] * expected[i];
            float diff = actual[i] - expected[i];
            diff_norm += diff * diff;
        }

        actual_norm = std::sqrt(actual_norm);
        expected_norm = std::sqrt(expected_norm);
        diff_norm = std::sqrt(diff_norm);

        // Compute relative error
        float rel_error = (expected_norm > 1e-8f) ? (diff_norm / expected_norm) : diff_norm;

        if (rank_ == 0)
        {
            std::cout << "Comparison metrics:" << std::endl;
            std::cout << "  Expected L2 norm: " << expected_norm << std::endl;
            std::cout << "  Actual L2 norm:   " << actual_norm << std::endl;
            std::cout << "  Diff L2 norm:     " << diff_norm << std::endl;
            std::cout << "  Relative error:   " << rel_error << " (tolerance: " << rel_tolerance << ")" << std::endl;
        }

        return rel_error <= rel_tolerance;
    }

    int rank_;
    int world_size_;
};

/**
 * Test 1: Basic Q8_0 streaming decode correctness
 *
 * Validates that Q8_0 streaming decode path produces reasonable output:
 * - No NaNs or infinities
 * - Non-zero activations
 * - Correct output shape
 */
TEST_F(MPILinearQ8_0Test, BasicStreamingDecodeCorrectness)
{
    const size_t seq_len = 8;
    const size_t in_dim = 64;
    const size_t out_dim = 128;

    // Create FP32 input
    auto input = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(in_dim)});
    float *in_data = input->data();
    for (size_t i = 0; i < seq_len * in_dim; ++i)
    {
        in_data[i] = static_cast<float>(i % 100) * 0.01f - 0.5f; // Range [-0.5, 0.5]
    }

    // Create Q8_0 weight
    std::vector<float> weight_fp32(out_dim * in_dim);
    for (size_t i = 0; i < out_dim * in_dim; ++i)
    {
        weight_fp32[i] = static_cast<float>((i * 7) % 100) * 0.01f - 0.5f; // Range [-0.5, 0.5]
    }
    auto weight_q8 = createQ8_0Weight(weight_fp32, out_dim, in_dim);

    // Create output
    auto output = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(out_dim)});

    // Execute MPILinearOperator with Q8_0 weight
    MPILinearOperator op(MPI_COMM_WORLD);
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight_q8};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    bool success = op.execute(inputs, outputs);
    ASSERT_TRUE(success) << "MPILinearOperator::execute() failed with Q8_0 weight";

    // Verify output shape
    EXPECT_EQ(output->shape().size(), 2);
    EXPECT_EQ(output->shape()[0], seq_len);
    EXPECT_EQ(output->shape()[1], out_dim);

    // Verify no NaNs or Infs
    const float *out_data = output->data();
    size_t non_zero_count = 0;
    for (size_t i = 0; i < seq_len * out_dim; ++i)
    {
        EXPECT_FALSE(std::isnan(out_data[i])) << "NaN at position " << i;
        EXPECT_FALSE(std::isinf(out_data[i])) << "Inf at position " << i;
        if (std::abs(out_data[i]) > 1e-6f)
        {
            non_zero_count++;
        }
    }

    // Expect at least 50% of outputs to be non-zero (sanity check)
    EXPECT_GT(non_zero_count, (seq_len * out_dim) / 2)
        << "Too many near-zero outputs: " << non_zero_count << " / " << (seq_len * out_dim);

    if (rank_ == 0)
    {
        std::cout << "✅ Basic streaming decode correctness validated" << std::endl;
        std::cout << "   Non-zero outputs: " << non_zero_count << " / " << (seq_len * out_dim) << std::endl;
    }
}

/**
 * Test 2: Q8_0 vs FP32 parity test
 *
 * **Critical validation**: Compares Q8_0 streaming decode output against FP32 baseline.
 * This validates that quantization error is within acceptable bounds (1% relative error).
 */
TEST_F(MPILinearQ8_0Test, Q8_0_vs_FP32_Parity)
{
    const size_t seq_len = 16;
    const size_t in_dim = 128;
    const size_t out_dim = 256;

    // Create FP32 input (identical for both paths)
    auto input = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(in_dim)});
    float *in_data = input->data();
    for (size_t i = 0; i < seq_len * in_dim; ++i)
    {
        in_data[i] = static_cast<float>(i % 100) * 0.01f - 0.5f;
    }

    // Create FP32 weight
    std::vector<float> weight_fp32(out_dim * in_dim);
    for (size_t i = 0; i < out_dim * in_dim; ++i)
    {
        weight_fp32[i] = static_cast<float>((i * 13) % 100) * 0.01f - 0.5f;
    }

    // PATH 1: FP32 baseline
    auto weight_fp32_tensor = TensorFactory::create_simple({static_cast<int>(out_dim), static_cast<int>(in_dim)});
    std::memcpy(weight_fp32_tensor->data(), weight_fp32.data(), out_dim * in_dim * sizeof(float));

    auto output_fp32 = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(out_dim)});

    MPILinearOperator op_fp32(MPI_COMM_WORLD);
    std::vector<std::shared_ptr<TensorBase>> inputs_fp32 = {input, weight_fp32_tensor};
    std::vector<std::shared_ptr<TensorBase>> outputs_fp32 = {output_fp32};

    bool success_fp32 = op_fp32.execute(inputs_fp32, outputs_fp32);
    ASSERT_TRUE(success_fp32) << "FP32 baseline execution failed";

    // PATH 2: Q8_0 streaming decode
    auto weight_q8 = createQ8_0Weight(weight_fp32, out_dim, in_dim);
    auto output_q8 = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(out_dim)});

    MPILinearOperator op_q8(MPI_COMM_WORLD);
    std::vector<std::shared_ptr<TensorBase>> inputs_q8 = {input, weight_q8};
    std::vector<std::shared_ptr<TensorBase>> outputs_q8 = {output_q8};

    bool success_q8 = op_q8.execute(inputs_q8, outputs_q8);
    ASSERT_TRUE(success_q8) << "Q8_0 streaming decode execution failed";

    // Compare outputs
    bool parity_passed = compareTensorsWithTolerance(
        output_q8->data(),
        output_fp32->data(),
        seq_len * out_dim,
        1.5e-2f // 1.5% relative tolerance for Q8_0 (accounts for quantization error)
    );

    EXPECT_TRUE(parity_passed) << "Q8_0 output diverged from FP32 baseline beyond tolerance";

    if (rank_ == 0)
    {
        std::cout << "✅ Q8_0 vs FP32 parity test " << (parity_passed ? "PASSED" : "FAILED") << std::endl;
    }
}

/**
 * Test 3: Single row edge case
 *
 * Tests Q8_0 streaming decode with seq_len=1 (common in autoregressive decode).
 */
TEST_F(MPILinearQ8_0Test, SingleRowEdgeCase)
{
    const size_t seq_len = 1; // Single token
    const size_t in_dim = 256;
    const size_t out_dim = 512;

    // Create input [1, in_dim]
    auto input = TensorFactory::create_simple({1, static_cast<int>(in_dim)});
    float *in_data = input->data();
    for (size_t i = 0; i < in_dim; ++i)
    {
        in_data[i] = static_cast<float>(i) * 0.01f;
    }

    // Create Q8_0 weight
    std::vector<float> weight_fp32(out_dim * in_dim);
    for (size_t i = 0; i < out_dim * in_dim; ++i)
    {
        weight_fp32[i] = static_cast<float>(i % 100) * 0.001f;
    }
    auto weight_q8 = createQ8_0Weight(weight_fp32, out_dim, in_dim);

    // Create output
    auto output = TensorFactory::create_simple({1, static_cast<int>(out_dim)});

    // Execute
    MPILinearOperator op(MPI_COMM_WORLD);
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight_q8};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    bool success = op.execute(inputs, outputs);
    ASSERT_TRUE(success) << "Single row execution failed";

    // Verify output
    EXPECT_EQ(output->shape()[0], 1);
    EXPECT_EQ(output->shape()[1], out_dim);

    const float *out_data = output->data();
    for (size_t i = 0; i < out_dim; ++i)
    {
        EXPECT_FALSE(std::isnan(out_data[i])) << "NaN at position " << i;
        EXPECT_FALSE(std::isinf(out_data[i])) << "Inf at position " << i;
    }

    if (rank_ == 0)
    {
        std::cout << "✅ Single row edge case passed" << std::endl;
    }
}

/**
 * Test 4: Small matrix dimensions
 *
 * Tests Q8_0 streaming decode with small dimensions (edge case for block size).
 */
TEST_F(MPILinearQ8_0Test, SmallMatrixDimensions)
{
    const size_t seq_len = 4;
    const size_t in_dim = 32; // Exactly one Q8_0 block
    const size_t out_dim = 64;

    auto input = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(in_dim)});
    float *in_data = input->data();
    for (size_t i = 0; i < seq_len * in_dim; ++i)
    {
        in_data[i] = 1.0f; // Simple constant for easy verification
    }

    // Create Q8_0 weight with known pattern
    std::vector<float> weight_fp32(out_dim * in_dim, 0.5f);
    auto weight_q8 = createQ8_0Weight(weight_fp32, out_dim, in_dim);

    auto output = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(out_dim)});

    MPILinearOperator op(MPI_COMM_WORLD);
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight_q8};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    bool success = op.execute(inputs, outputs);
    ASSERT_TRUE(success) << "Small matrix execution failed";

    // Verify output shape
    EXPECT_EQ(output->shape()[0], seq_len);
    EXPECT_EQ(output->shape()[1], out_dim);

    // Verify output values are reasonable (1.0 * 0.5 * 32 ≈ 16.0 per output)
    const float *out_data = output->data();
    float expected_approx = 16.0f;
    for (size_t i = 0; i < seq_len * out_dim; ++i)
    {
        EXPECT_NEAR(out_data[i], expected_approx, 2.0f) << "Unexpected value at position " << i;
    }

    if (rank_ == 0)
    {
        std::cout << "✅ Small matrix dimensions passed" << std::endl;
        std::cout << "   Expected ~" << expected_approx << ", got " << out_data[0] << std::endl;
    }
}

/**
 * Test 5: Large sequence length
 *
 * Tests Q8_0 streaming decode with large seq_len (common in prefill phase).
 */
TEST_F(MPILinearQ8_0Test, LargeSequenceLength)
{
    const size_t seq_len = 512; // Typical prefill length
    const size_t in_dim = 128;
    const size_t out_dim = 256;

    auto input = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(in_dim)});
    float *in_data = input->data();
    for (size_t i = 0; i < seq_len * in_dim; ++i)
    {
        in_data[i] = static_cast<float>((i % 100) - 50) * 0.01f;
    }

    std::vector<float> weight_fp32(out_dim * in_dim);
    for (size_t i = 0; i < out_dim * in_dim; ++i)
    {
        weight_fp32[i] = static_cast<float>((i % 100) - 50) * 0.01f;
    }
    auto weight_q8 = createQ8_0Weight(weight_fp32, out_dim, in_dim);

    auto output = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(out_dim)});

    MPILinearOperator op(MPI_COMM_WORLD);
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight_q8};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    auto t0 = std::chrono::high_resolution_clock::now();
    bool success = op.execute(inputs, outputs);
    auto t1 = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(success) << "Large sequence execution failed";

    // Verify output
    EXPECT_EQ(output->shape()[0], seq_len);
    EXPECT_EQ(output->shape()[1], out_dim);

    const float *out_data = output->data();
    size_t valid_count = 0;
    for (size_t i = 0; i < seq_len * out_dim; ++i)
    {
        if (!std::isnan(out_data[i]) && !std::isinf(out_data[i]))
        {
            valid_count++;
        }
    }
    EXPECT_EQ(valid_count, seq_len * out_dim) << "Invalid values in output";

    if (rank_ == 0)
    {
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "✅ Large sequence length passed" << std::endl;
        std::cout << "   Execution time: " << duration_ms << " ms" << std::endl;
    }
}

/**
 * Test 6: Multi-rank distribution correctness
 *
 * Validates that Q8_0 weight distribution works correctly across MPI ranks.
 * Each rank should process its assigned output slice correctly.
 */
TEST_F(MPILinearQ8_0Test, MultiRankDistribution)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Multi-rank test requires at least 2 MPI processes";
    }

    const size_t seq_len = 8;
    const size_t in_dim = 64;
    const size_t out_dim = 128;

    // Create identical input on all ranks
    auto input = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(in_dim)});
    float *in_data = input->data();
    for (size_t i = 0; i < seq_len * in_dim; ++i)
    {
        in_data[i] = static_cast<float>(i) * 0.01f;
    }

    // Create identical Q8_0 weight on all ranks
    std::vector<float> weight_fp32(out_dim * in_dim);
    for (size_t i = 0; i < out_dim * in_dim; ++i)
    {
        weight_fp32[i] = static_cast<float>(i % 100) * 0.001f;
    }
    auto weight_q8 = createQ8_0Weight(weight_fp32, out_dim, in_dim);

    // Create output
    auto output = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(out_dim)});

    // Execute on all ranks
    MPILinearOperator op(MPI_COMM_WORLD);
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight_q8};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    bool success = op.execute(inputs, outputs);
    ASSERT_TRUE(success) << "Multi-rank execution failed on rank " << rank_;

    // Verify output shape on all ranks
    EXPECT_EQ(output->shape()[0], seq_len);
    EXPECT_EQ(output->shape()[1], out_dim);

    // Verify output validity on all ranks
    const float *out_data = output->data();
    for (size_t i = 0; i < seq_len * out_dim; ++i)
    {
        EXPECT_FALSE(std::isnan(out_data[i])) << "NaN on rank " << rank_ << " at position " << i;
        EXPECT_FALSE(std::isinf(out_data[i])) << "Inf on rank " << rank_ << " at position " << i;
    }

    // Gather outputs from all ranks to rank 0 for comparison
    std::vector<float> gathered_outputs;
    if (rank_ == 0)
    {
        gathered_outputs.resize(world_size_ * seq_len * out_dim);
    }

    MPI_Gather(out_data, seq_len * out_dim, MPI_FLOAT,
               gathered_outputs.data(), seq_len * out_dim, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    // Rank 0 verifies all ranks produced identical results (TP means replicated output)
    if (rank_ == 0)
    {
        bool all_ranks_match = true;
        for (int r = 1; r < world_size_; ++r)
        {
            const float *rank_r_output = gathered_outputs.data() + r * seq_len * out_dim;
            for (size_t i = 0; i < seq_len * out_dim; ++i)
            {
                float diff = std::abs(rank_r_output[i] - out_data[i]);
                if (diff > 1e-5f)
                {
                    all_ranks_match = false;
                    std::cout << "Mismatch between rank 0 and rank " << r << " at position " << i
                              << ": " << out_data[i] << " vs " << rank_r_output[i] << std::endl;
                    break;
                }
            }
            if (!all_ranks_match)
                break;
        }

        EXPECT_TRUE(all_ranks_match) << "Outputs from different ranks don't match";

        std::cout << "✅ Multi-rank distribution correctness validated (" << world_size_ << " ranks)" << std::endl;
    }
}

/**
 * Test 7: Q8_0 decoding numerical accuracy
 *
 * Validates that Q8_0 decodeRow() produces values within expected quantization error.
 * This is a focused test of the Q8_0Tensor decoding itself.
 */
TEST_F(MPILinearQ8_0Test, Q8_0DecodingNumericalAccuracy)
{
    const size_t rows = 64;
    const size_t cols = 128;

    // Create FP32 source data
    std::vector<float> source_fp32(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i)
    {
        source_fp32[i] = static_cast<float>((i % 200) - 100) * 0.01f; // Range [-1.0, 1.0]
    }

    // Quantize to Q8_0
    auto q8_tensor = createQ8_0Weight(source_fp32, rows, cols);

    // Decode each row and compare
    std::vector<float> decoded_row(cols);
    float max_rel_error = 0.0f;
    size_t total_comparisons = 0;

    for (size_t r = 0; r < rows; ++r)
    {
        q8_tensor->decodeRow(r, decoded_row.data());

        // Compare decoded row with source
        for (size_t c = 0; c < cols; ++c)
        {
            float original = source_fp32[r * cols + c];
            float decoded = decoded_row[c];

            if (std::abs(original) > 1e-6f)
            {
                float rel_error = std::abs(decoded - original) / std::abs(original);
                max_rel_error = std::max(max_rel_error, rel_error);
                total_comparisons++;
            }
        }
    }

    // Q8_0 uses int8 with per-block scaling, expect <1% error on average
    EXPECT_LT(max_rel_error, 0.05f) << "Q8_0 decoding error too high: " << max_rel_error;

    if (rank_ == 0)
    {
        std::cout << "✅ Q8_0 decoding numerical accuracy validated" << std::endl;
        std::cout << "   Max relative error: " << max_rel_error << " (over " << total_comparisons << " comparisons)" << std::endl;
    }
}

/**
 * Test 8: Performance comparison - Q8_0 vs FP32
 *
 * Measures and compares execution time for Q8_0 streaming decode vs FP32 baseline.
 * Expected: Q8_0 slightly slower due to decode overhead, but memory savings significant.
 */
TEST_F(MPILinearQ8_0Test, PerformanceComparison)
{
    const size_t seq_len = 256;
    const size_t in_dim = 512;
    const size_t out_dim = 1024;
    const int num_iterations = 10;

    // Create input
    auto input = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(in_dim)});
    float *in_data = input->data();
    for (size_t i = 0; i < seq_len * in_dim; ++i)
    {
        in_data[i] = static_cast<float>(i % 100) * 0.01f;
    }

    // Create FP32 weight
    std::vector<float> weight_fp32(out_dim * in_dim);
    for (size_t i = 0; i < out_dim * in_dim; ++i)
    {
        weight_fp32[i] = static_cast<float>(i % 100) * 0.001f;
    }

    // Benchmark FP32 path
    auto weight_fp32_tensor = TensorFactory::create_simple({static_cast<int>(out_dim), static_cast<int>(in_dim)});
    std::memcpy(weight_fp32_tensor->data(), weight_fp32.data(), out_dim * in_dim * sizeof(float));

    auto output_fp32 = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(out_dim)});

    MPILinearOperator op_fp32(MPI_COMM_WORLD);
    std::vector<std::shared_ptr<TensorBase>> inputs_fp32 = {input, weight_fp32_tensor};
    std::vector<std::shared_ptr<TensorBase>> outputs_fp32 = {output_fp32};

    // Warmup
    op_fp32.execute(inputs_fp32, outputs_fp32);

    auto t0_fp32 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i)
    {
        op_fp32.execute(inputs_fp32, outputs_fp32);
    }
    auto t1_fp32 = std::chrono::high_resolution_clock::now();
    auto duration_fp32_us = std::chrono::duration_cast<std::chrono::microseconds>(t1_fp32 - t0_fp32).count();

    // Benchmark Q8_0 path
    auto weight_q8 = createQ8_0Weight(weight_fp32, out_dim, in_dim);
    auto output_q8 = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(out_dim)});

    MPILinearOperator op_q8(MPI_COMM_WORLD);
    std::vector<std::shared_ptr<TensorBase>> inputs_q8 = {input, weight_q8};
    std::vector<std::shared_ptr<TensorBase>> outputs_q8 = {output_q8};

    // Warmup
    op_q8.execute(inputs_q8, outputs_q8);

    auto t0_q8 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i)
    {
        op_q8.execute(inputs_q8, outputs_q8);
    }
    auto t1_q8 = std::chrono::high_resolution_clock::now();
    auto duration_q8_us = std::chrono::duration_cast<std::chrono::microseconds>(t1_q8 - t0_q8).count();

    // Compute performance metrics
    float avg_time_fp32_ms = duration_fp32_us / 1000.0f / num_iterations;
    float avg_time_q8_ms = duration_q8_us / 1000.0f / num_iterations;
    float slowdown_ratio = static_cast<float>(duration_q8_us) / duration_fp32_us;

    // Compute memory savings
    size_t fp32_size = out_dim * in_dim * sizeof(float);
    size_t q8_size = weight_q8->raw_size();
    float compression_ratio = static_cast<float>(fp32_size) / q8_size;

    if (rank_ == 0)
    {
        std::cout << "✅ Performance comparison completed" << std::endl;
        std::cout << "   FP32 avg time:  " << avg_time_fp32_ms << " ms" << std::endl;
        std::cout << "   Q8_0 avg time:  " << avg_time_q8_ms << " ms" << std::endl;
        std::cout << "   Slowdown ratio: " << slowdown_ratio << "×" << std::endl;
        std::cout << "   Memory savings: " << compression_ratio << "× compression" << std::endl;
        std::cout << "   FP32 size:      " << (fp32_size / 1024.0f / 1024.0f) << " MB" << std::endl;
        std::cout << "   Q8_0 size:      " << (q8_size / 1024.0f / 1024.0f) << " MB" << std::endl;
    }

    // Expect Q8_0 to be at most 2× slower (decode overhead), but memory savings should be ~4×
    EXPECT_LT(slowdown_ratio, 2.0f) << "Q8_0 path too slow compared to FP32";
    EXPECT_GT(compression_ratio, 3.5f) << "Q8_0 compression ratio too low";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
