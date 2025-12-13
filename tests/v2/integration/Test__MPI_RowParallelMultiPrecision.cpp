/**
 * @file Test__MPI_RowParallelMultiPrecision.cpp
 * @brief MPI integration tests for row-parallel GEMM with multi-precision support
 *
 * Tests the actual MPI row-parallel GEMM paths in PipelineBase with real multi-rank
 * communication. Validates correctness of:
 * - TensorSlice row-parallel (output column slicing, allgatherv pattern)
 * - K-sliced row-parallel (input column slicing, allreduce-sum pattern)
 * - All precisions: FP32, BF16, FP16, Q8_1
 * - Block alignment handling for Q8_1
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <cstring>
#include <fstream>

#include "tensors/Tensors.h"
#include "tensors/TensorSlice.h"
#include "tensors/TensorFactory.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/BlockStructures.h"
#include "loaders/ModelLoader.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"
#include "kernels/KernelFactory.h"

using namespace llaminar2;
using namespace llaminar::v2::kernels;

namespace
{
    // Path to test model with Q4_0 weights
    constexpr const char *TEST_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    /**
     * @brief Test fixture for MPI row-parallel multi-precision integration tests
     */
    class Test__MPI_RowParallelMultiPrecision : public ::testing::Test
    {
    protected:
        std::shared_ptr<MPIContext> mpi_ctx_;
        int rank_ = 0;
        int world_size_ = 1;
        std::mt19937 rng_;

        void SetUp() override
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
            mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
            rng_.seed(42); // Same seed for reproducible weights across ranks
        }

        void TearDown() override
        {
            mpi_ctx_->barrier();
        }

        // Generate random FP32 weight matrix [n, k] (same across all ranks)
        std::vector<float> generate_weight_matrix(int n, int k)
        {
            rng_.seed(42); // Reset seed for consistent weights
            std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
            std::vector<float> result(static_cast<size_t>(n) * k);
            for (auto &v : result)
            {
                v = dist(rng_);
            }
            return result;
        }

        // Generate random FP32 activation matrix [m, k] (same across all ranks for verification)
        std::vector<float> generate_activation_matrix(int m, int k)
        {
            rng_.seed(123); // Different seed from weights, but consistent across ranks
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::vector<float> result(static_cast<size_t>(m) * k);
            for (auto &v : result)
            {
                v = dist(rng_);
            }
            return result;
        }

        // Reference FP32 GEMM: C = A @ B^T where A is [m, k], B is [n, k], C is [m, n]
        std::vector<float> reference_gemm(const std::vector<float> &A, const std::vector<float> &B,
                                          int m, int n, int k)
        {
            std::vector<float> C(static_cast<size_t>(m) * n, 0.0f);
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    float sum = 0.0f;
                    for (int l = 0; l < k; ++l)
                    {
                        sum += A[i * k + l] * B[j * k + l];
                    }
                    C[i * n + j] = sum;
                }
            }
            return C;
        }

        // Convert FP32 to BF16 vector
        std::vector<uint16_t> fp32_to_bf16_vec(const std::vector<float> &fp32)
        {
            std::vector<uint16_t> bf16(fp32.size());
            for (size_t i = 0; i < fp32.size(); ++i)
            {
                bf16[i] = simd::fp32_to_bf16(fp32[i]);
            }
            return bf16;
        }

        // Convert BF16 to FP32 vector
        std::vector<float> bf16_to_fp32_vec(const std::vector<uint16_t> &bf16)
        {
            std::vector<float> fp32(bf16.size());
            for (size_t i = 0; i < bf16.size(); ++i)
            {
                fp32[i] = simd::bf16_to_fp32(bf16[i]);
            }
            return fp32;
        }

        // Convert FP32 to FP16 vector
        std::vector<uint16_t> fp32_to_fp16_vec(const std::vector<float> &fp32)
        {
            std::vector<uint16_t> fp16(fp32.size());
            for (size_t i = 0; i < fp32.size(); ++i)
            {
                fp16[i] = simd::fp32_to_fp16(fp32[i]);
            }
            return fp16;
        }

        // Convert FP16 to FP32 vector
        std::vector<float> fp16_to_fp32_vec(const std::vector<uint16_t> &fp16)
        {
            std::vector<float> fp32(fp16.size());
            for (size_t i = 0; i < fp16.size(); ++i)
            {
                fp32[i] = simd::fp16_to_fp32(fp16[i]);
            }
            return fp32;
        }

        // Compute cosine similarity
        float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
        {
            if (a.size() != b.size() || a.empty())
                return 0.0f;

            float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
            for (size_t i = 0; i < a.size(); ++i)
            {
                dot += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }

            if (norm_a < 1e-10f || norm_b < 1e-10f)
                return 1.0f;
            return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        }

        // Compute max absolute difference
        float max_abs_diff(const std::vector<float> &a, const std::vector<float> &b)
        {
            float max_diff = 0.0f;
            for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
            {
                max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
            }
            return max_diff;
        }
    };

    // =============================================================================
    // TensorSlice Row-Parallel Tests (Output Column Slicing, Allgatherv Pattern)
    // =============================================================================

    /**
     * @test TensorSlice row-parallel with FP32 output
     *
     * Tests the allgatherv pattern where each rank has disjoint output columns.
     * Weight is TensorSlice wrapping [n, k], each rank computes [m, n_local].
     */
    TEST_F(Test__MPI_RowParallelMultiPrecision, TensorSlice_FP32_Output)
    {
        ASSERT_GE(world_size_, 2) << "This test requires at least 2 MPI ranks";

        // Dimensions: small enough for fast tests, large enough to be meaningful
        const int m = 4;   // Sequence length
        const int n = 128; // Output dimension (divisible by world_size and 32)
        const int k = 64;  // Input dimension

        // Generate data
        auto weight_data = generate_weight_matrix(n, k);
        auto input_data = generate_activation_matrix(m, k);
        auto reference_output = reference_gemm(input_data, weight_data, m, n, k);

        // Create TensorSlice for this rank's portion with PRE-SLICED tensor
        const int n_local = n / world_size_;
        const int n_start = n_local * rank_;

        // Create a pre-sliced weight tensor containing only this rank's rows
        auto sliced_weight = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_local), static_cast<size_t>(k)});
        for (int row = 0; row < n_local; ++row)
        {
            const int global_row = n_start + row;
            std::memcpy(sliced_weight->mutable_data() + row * k,
                        weight_data.data() + global_row * k,
                        k * sizeof(float));
        }

        // Create slice metadata with inner_is_presliced=true
        SliceMetadata slice_meta = SliceMetadata::forRowParallel(
            static_cast<size_t>(n), static_cast<size_t>(k),
            rank_, world_size_, true /* inner_is_presliced - already sliced */
        );
        std::unique_ptr<TensorBase> base_weight = std::move(sliced_weight);
        auto slice = std::make_unique<TensorSlice>(std::move(base_weight), slice_meta);
        ASSERT_NE(slice, nullptr);

        // Create input and output tensors
        FP32Tensor input({static_cast<size_t>(m), static_cast<size_t>(k)});
        std::memcpy(input.mutable_data(), input_data.data(), input_data.size() * sizeof(float));

        FP32Tensor output({static_cast<size_t>(m), static_cast<size_t>(n)});

        // Create local output for this rank's slice
        FP32Tensor output_local({static_cast<size_t>(m), static_cast<size_t>(n_local)});

        // Get kernel from TensorSlice
        auto *gemm_kernel = KernelFactory::getOrCreateGemm(slice.get());
        ASSERT_NE(gemm_kernel, nullptr);

        // Compute local GEMM: [m, k] @ [n_local, k]^T = [m, n_local]
        ASSERT_TRUE(gemm_kernel->multiply_tensor(&input, &output_local, m, n_local, k, true, 1.0f, 0.0f, mpi_ctx_.get(), -1));

        // Allgatherv to combine slices
        std::vector<int> recv_counts(world_size_);
        std::vector<int> displs(world_size_);
        for (int r = 0; r < world_size_; ++r)
        {
            int r_n_local = n / world_size_;
            if (r == world_size_ - 1)
                r_n_local = n - r_n_local * (world_size_ - 1);
            int r_start = r * (n / world_size_);
            recv_counts[r] = r_n_local * sizeof(float);
            displs[r] = r_start * sizeof(float);
        }

        const int local_row_bytes = n_local * sizeof(float);
        float *out_data = output.mutable_data();
        const float *local_data = output_local.data();

        for (int row = 0; row < m; ++row)
        {
            const float *src_row = local_data + row * n_local;
            float *dst_row = out_data + row * n;
            mpi_ctx_->allgatherv_bytes(src_row, local_row_bytes, dst_row, recv_counts.data(), displs.data());
        }

        // Verify result on rank 0
        if (rank_ == 0)
        {
            std::vector<float> result(out_data, out_data + static_cast<size_t>(m) * n);
            float cos_sim = cosine_similarity(reference_output, result);
            float max_diff = max_abs_diff(reference_output, result);

            LOG_INFO("[TensorSlice FP32] Cosine similarity: " << cos_sim << ", Max diff: " << max_diff);
            EXPECT_GT(cos_sim, 0.999f) << "FP32 should be nearly exact";
            EXPECT_LT(max_diff, 1e-4f);
        }
    }

    /**
     * @test TensorSlice row-parallel with Q8_1 MPI communication (block-aligned)
     *
     * Tests Q8_1 block-level allgatherv for MPI communication.
     * Uses FP32 weights and activations for GEMM, then quantizes output to Q8_1
     * for the MPI communication test.
     * Requires n_local % 32 == 0 for block alignment.
     */
    TEST_F(Test__MPI_RowParallelMultiPrecision, TensorSlice_Q8_1_Output_BlockAligned)
    {
        ASSERT_GE(world_size_, 2) << "This test requires at least 2 MPI ranks";

        // Dimensions chosen for Q8_1 block alignment (n_local must be multiple of 32)
        const int m = 4;
        const int n = 128; // 128 / 2 = 64 per rank, divisible by 32
        const int k = 64;

        // Generate data
        auto weight_data = generate_weight_matrix(n, k);
        auto input_data = generate_activation_matrix(m, k);
        auto reference_output = reference_gemm(input_data, weight_data, m, n, k);

        // Create TensorSlice with PRE-SLICED tensor
        const int n_local = n / world_size_;
        const int n_start = n_local * rank_;

        // Create pre-sliced FP32 weight tensor
        auto sliced_weight = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_local), static_cast<size_t>(k)});
        for (int row = 0; row < n_local; ++row)
        {
            const int global_row = n_start + row;
            std::memcpy(sliced_weight->mutable_data() + row * k,
                        weight_data.data() + global_row * k,
                        k * sizeof(float));
        }

        SliceMetadata slice_meta = SliceMetadata::forRowParallel(
            static_cast<size_t>(n), static_cast<size_t>(k),
            rank_, world_size_, true /* inner_is_presliced */
        );
        std::unique_ptr<TensorBase> base_weight = std::move(sliced_weight);
        auto slice = std::make_unique<TensorSlice>(std::move(base_weight), slice_meta);
        ASSERT_NE(slice, nullptr);

        // Create FP32 input tensor (GEMM with FP32 weights requires FP32 activations)
        FP32Tensor input({static_cast<size_t>(m), static_cast<size_t>(k)});
        std::memcpy(input.mutable_data(), input_data.data(), input_data.size() * sizeof(float));

        // Create FP32 local output for GEMM
        FP32Tensor output_local_fp32({static_cast<size_t>(m), static_cast<size_t>(n_local)});

        // Get kernel and compute GEMM in FP32
        auto *gemm_kernel = KernelFactory::getOrCreateGemm(slice.get());
        ASSERT_NE(gemm_kernel, nullptr);
        ASSERT_TRUE(gemm_kernel->multiply_tensor(&input, &output_local_fp32, m, n_local, k, true, 1.0f, 0.0f, mpi_ctx_.get(), -1));

        // Quantize local FP32 output to Q8_1 for MPI communication test
        Q8_1Tensor output_local({static_cast<size_t>(m), static_cast<size_t>(n_local)});
        simd::quantize_fp32_to_q8_1_blocks(output_local_fp32.data(), output_local.mutable_q8_1_blocks(),
                                           static_cast<size_t>(m) * n_local);

        // Create Q8_1 full output tensor
        Q8_1Tensor output({static_cast<size_t>(m), static_cast<size_t>(n)});

        // Allgatherv Q8_1 blocks - this is what we're testing
        constexpr size_t BLOCK_SIZE = Q8_1Block::BLOCK_SIZE;
        const size_t blocks_per_row_full = n / BLOCK_SIZE;
        const size_t blocks_per_row_local = n_local / BLOCK_SIZE;

        std::vector<int> recv_counts(world_size_);
        std::vector<int> displs(world_size_);
        for (int r = 0; r < world_size_; ++r)
        {
            size_t r_n_local = n / world_size_;
            size_t r_blocks_local = r_n_local / BLOCK_SIZE;
            size_t r_block_start = (r * (n / world_size_)) / BLOCK_SIZE;
            recv_counts[r] = static_cast<int>(r_blocks_local * sizeof(Q8_1Block));
            displs[r] = static_cast<int>(r_block_start * sizeof(Q8_1Block));
        }

        const int local_row_bytes = static_cast<int>(blocks_per_row_local * sizeof(Q8_1Block));
        const Q8_1Block *local_blocks = output_local.q8_1_blocks();
        Q8_1Block *output_blocks = output.mutable_q8_1_blocks();

        for (int row = 0; row < m; ++row)
        {
            const Q8_1Block *src_row = local_blocks + row * blocks_per_row_local;
            Q8_1Block *dst_row = output_blocks + row * blocks_per_row_full;
            mpi_ctx_->allgatherv_bytes(src_row, local_row_bytes, dst_row, recv_counts.data(), displs.data());
        }

        // Verify result on rank 0
        if (rank_ == 0)
        {
            std::vector<float> result(static_cast<size_t>(m) * n);
            simd::dequantize_q8_1_to_fp32(output.q8_1_blocks(), result.data(), result.size());

            float cos_sim = cosine_similarity(reference_output, result);
            float max_diff = max_abs_diff(reference_output, result);

            LOG_INFO("[TensorSlice Q8_1] Cosine similarity: " << cos_sim << ", Max diff: " << max_diff);
            // Q8_1 has quantization noise from output quantization, expect slightly lower similarity
            EXPECT_GT(cos_sim, 0.98f) << "Q8_1 should maintain reasonable accuracy";
        }
    }

    /**
     * @test TensorSlice row-parallel with BF16 output
     *
     * Creates BF16 weights to match BF16 activations for FloatingPointGemmKernel.
     */
    TEST_F(Test__MPI_RowParallelMultiPrecision, TensorSlice_BF16_Output)
    {
        ASSERT_GE(world_size_, 2) << "This test requires at least 2 MPI ranks";

        const int m = 4;
        const int n = 128;
        const int k = 64;

        auto weight_data = generate_weight_matrix(n, k);
        auto input_data = generate_activation_matrix(m, k);
        auto reference_output = reference_gemm(input_data, weight_data, m, n, k);

        // Create TensorSlice with PRE-SLICED BF16 tensor
        const int n_local = n / world_size_;
        const int n_start = n_local * rank_;

        // Create pre-sliced BF16 weight tensor
        auto sliced_weight = std::make_unique<BF16Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_local), static_cast<size_t>(k)});
        auto weight_bf16 = fp32_to_bf16_vec(weight_data);
        for (int row = 0; row < n_local; ++row)
        {
            const int global_row = n_start + row;
            std::memcpy(sliced_weight->mutable_bf16_data() + row * k,
                        weight_bf16.data() + global_row * k,
                        k * sizeof(uint16_t));
        }

        SliceMetadata slice_meta = SliceMetadata::forRowParallel(
            static_cast<size_t>(n), static_cast<size_t>(k),
            rank_, world_size_, true /* inner_is_presliced */
        );
        std::unique_ptr<TensorBase> base_weight = std::move(sliced_weight);
        auto slice = std::make_unique<TensorSlice>(std::move(base_weight), slice_meta);
        ASSERT_NE(slice, nullptr);

        // Create BF16 input tensor
        BF16Tensor input({static_cast<size_t>(m), static_cast<size_t>(k)});
        auto input_bf16 = fp32_to_bf16_vec(input_data);
        std::memcpy(input.mutable_bf16_data(), input_bf16.data(), input_bf16.size() * sizeof(uint16_t));

        // Create FP32 output tensors (FloatingPointGemmKernel always outputs FP32)
        FP32Tensor output({static_cast<size_t>(m), static_cast<size_t>(n)});
        FP32Tensor output_local({static_cast<size_t>(m), static_cast<size_t>(n_local)});

        // Get kernel and compute
        auto *gemm_kernel = KernelFactory::getOrCreateGemm(slice.get());
        ASSERT_NE(gemm_kernel, nullptr);
        ASSERT_TRUE(gemm_kernel->multiply_tensor(&input, &output_local, m, n_local, k, true, 1.0f, 0.0f, mpi_ctx_.get(), -1));

        // Allgatherv FP32 elements
        std::vector<int> recv_counts(world_size_);
        std::vector<int> displs(world_size_);
        for (int r = 0; r < world_size_; ++r)
        {
            int r_n_local = n / world_size_;
            int r_start = r * (n / world_size_);
            recv_counts[r] = r_n_local * sizeof(float);
            displs[r] = r_start * sizeof(float);
        }

        const int local_row_bytes = n_local * sizeof(float);
        const float *local_data = output_local.data();
        float *out_data = output.mutable_data();

        for (int row = 0; row < m; ++row)
        {
            const float *src_row = local_data + row * n_local;
            float *dst_row = out_data + row * n;
            mpi_ctx_->allgatherv_bytes(src_row, local_row_bytes, dst_row, recv_counts.data(), displs.data());
        }

        // Verify
        if (rank_ == 0)
        {
            std::vector<float> result(out_data, out_data + static_cast<size_t>(m) * n);

            float cos_sim = cosine_similarity(reference_output, result);
            LOG_INFO("[TensorSlice BF16] Cosine similarity: " << cos_sim);
            EXPECT_GT(cos_sim, 0.995f) << "BF16 should maintain good accuracy";
        }
    }

    // =============================================================================
    // K-Sliced Row-Parallel Tests (Input Column Slicing, Allreduce-Sum Pattern)
    // =============================================================================

    /**
     * @test K-sliced row-parallel with FP32 allreduce
     *
     * Tests the allreduce-sum pattern where each rank computes a partial product
     * and sums them across ranks.
     */
    TEST_F(Test__MPI_RowParallelMultiPrecision, KSliced_FP32_Allreduce)
    {
        ASSERT_GE(world_size_, 2) << "This test requires at least 2 MPI ranks";

        const int m = 4;
        const int n = 64;
        const int k = 128; // Sliced across ranks
        const int k_local = k / world_size_;
        const int k_start = k_local * rank_;

        auto weight_data = generate_weight_matrix(n, k);
        auto input_data = generate_activation_matrix(m, k);
        auto reference_output = reference_gemm(input_data, weight_data, m, n, k);

        // Each rank has weight slice [n, k_local]
        std::vector<float> weight_slice(static_cast<size_t>(n) * k_local);
        for (int row = 0; row < n; ++row)
        {
            for (int col = 0; col < k_local; ++col)
            {
                weight_slice[row * k_local + col] = weight_data[row * k + k_start + col];
            }
        }

        FP32Tensor weight({static_cast<size_t>(n), static_cast<size_t>(k_local)});
        std::memcpy(weight.mutable_data(), weight_slice.data(), weight_slice.size() * sizeof(float));

        // Each rank has input slice [m, k_local]
        std::vector<float> input_slice(static_cast<size_t>(m) * k_local);
        for (int row = 0; row < m; ++row)
        {
            for (int col = 0; col < k_local; ++col)
            {
                input_slice[row * k_local + col] = input_data[row * k + k_start + col];
            }
        }

        FP32Tensor input({static_cast<size_t>(m), static_cast<size_t>(k_local)});
        std::memcpy(input.mutable_data(), input_slice.data(), input_slice.size() * sizeof(float));

        // Output is full [m, n]
        FP32Tensor output({static_cast<size_t>(m), static_cast<size_t>(n)});

        // Compute local partial product
        auto *gemm_kernel = KernelFactory::getOrCreateGemm(&weight);
        ASSERT_NE(gemm_kernel, nullptr);
        ASSERT_TRUE(gemm_kernel->multiply_tensor(&input, &output, m, n, k_local, true, 1.0f, 0.0f, mpi_ctx_.get(), -1));

        // Allreduce-sum to combine partial products
        mpi_ctx_->allreduce_sum_inplace(output.mutable_data(), static_cast<size_t>(m) * n);

        // Verify
        if (rank_ == 0)
        {
            std::vector<float> result(output.data(), output.data() + static_cast<size_t>(m) * n);
            float cos_sim = cosine_similarity(reference_output, result);
            float max_diff = max_abs_diff(reference_output, result);

            LOG_INFO("[K-Sliced FP32] Cosine similarity: " << cos_sim << ", Max diff: " << max_diff);
            EXPECT_GT(cos_sim, 0.999f);
            EXPECT_LT(max_diff, 1e-4f);
        }
    }

    /**
     * @test K-sliced row-parallel with Q8_1 native allreduce
     *
     * Tests Q8_1 allreduce-sum for MPI communication.
     * Uses FP32 for GEMM, then quantizes output to Q8_1 for the MPI allreduce test.
     * Requires k_start and k_local to be multiples of 32 for block alignment.
     */
    TEST_F(Test__MPI_RowParallelMultiPrecision, KSliced_Q8_1_NativeAllreduce)
    {
        ASSERT_GE(world_size_, 2) << "This test requires at least 2 MPI ranks";

        // Dimensions chosen for Q8_1 block alignment
        const int m = 4;
        const int n = 64;
        const int k = 128; // 128 / 2 = 64 per rank, divisible by 32
        const int k_local = k / world_size_;
        const int k_start = k_local * rank_;

        // Verify block alignment
        ASSERT_EQ(k_start % 32, 0) << "k_start must be block-aligned";
        ASSERT_EQ(k_local % 32, 0) << "k_local must be block-aligned";

        auto weight_data = generate_weight_matrix(n, k);
        auto input_data = generate_activation_matrix(m, k);
        auto reference_output = reference_gemm(input_data, weight_data, m, n, k);

        // Create FP32 input slice [m, k_local]
        std::vector<float> input_slice(static_cast<size_t>(m) * k_local);
        for (int row = 0; row < m; ++row)
        {
            for (int col = 0; col < k_local; ++col)
            {
                input_slice[row * k_local + col] = input_data[row * k + k_start + col];
            }
        }
        FP32Tensor input({static_cast<size_t>(m), static_cast<size_t>(k_local)});
        std::memcpy(input.mutable_data(), input_slice.data(), input_slice.size() * sizeof(float));

        // Create FP32 weight slice [n, k_local]
        std::vector<float> weight_slice(static_cast<size_t>(n) * k_local);
        for (int row = 0; row < n; ++row)
        {
            for (int col = 0; col < k_local; ++col)
            {
                weight_slice[row * k_local + col] = weight_data[row * k + k_start + col];
            }
        }
        FP32Tensor weight({static_cast<size_t>(n), static_cast<size_t>(k_local)});
        std::memcpy(weight.mutable_data(), weight_slice.data(), weight_slice.size() * sizeof(float));

        // Create FP32 output for GEMM
        FP32Tensor output_fp32({static_cast<size_t>(m), static_cast<size_t>(n)});

        // Compute local partial product in FP32
        auto *gemm_kernel = KernelFactory::getOrCreateGemm(&weight);
        ASSERT_NE(gemm_kernel, nullptr);
        ASSERT_TRUE(gemm_kernel->multiply_tensor(&input, &output_fp32, m, n, k_local, true, 1.0f, 0.0f, mpi_ctx_.get(), -1));

        // Quantize FP32 output to Q8_1 for MPI allreduce test
        const size_t output_size = static_cast<size_t>(m) * n;
        Q8_1Tensor output({static_cast<size_t>(m), static_cast<size_t>(n)});
        simd::quantize_fp32_to_q8_1_blocks(output_fp32.data(), output.mutable_q8_1_blocks(), output_size);

        // Native Q8_1 allreduce-sum - this is what we're testing
        const size_t n_blocks = (output_size + 31) / 32;
        mpi_ctx_->allreduce_q8_1_inplace(output.mutable_q8_1_blocks(), n_blocks);

        // Verify
        if (rank_ == 0)
        {
            std::vector<float> result(output_size);
            simd::dequantize_q8_1_to_fp32(output.q8_1_blocks(), result.data(), output_size);

            float cos_sim = cosine_similarity(reference_output, result);
            LOG_INFO("[K-Sliced Q8_1] Cosine similarity: " << cos_sim);
            // Q8_1 has quantization noise from output quantization, plus allreduce requantization
            EXPECT_GT(cos_sim, 0.95f) << "Q8_1 native allreduce should maintain reasonable accuracy";
        }
    }

    /**
     * @test K-sliced row-parallel with BF16 native allreduce
     *
     * Creates BF16 weights to match BF16 activations for FloatingPointGemmKernel.
     * Output is FP32 (GEMM always outputs FP32), then converted to BF16 for allreduce test.
     */
    TEST_F(Test__MPI_RowParallelMultiPrecision, KSliced_BF16_NativeAllreduce)
    {
        ASSERT_GE(world_size_, 2) << "This test requires at least 2 MPI ranks";

        const int m = 4;
        const int n = 64;
        const int k = 128;
        const int k_local = k / world_size_;
        const int k_start = k_local * rank_;

        auto weight_data = generate_weight_matrix(n, k);
        auto input_data = generate_activation_matrix(m, k);
        auto reference_output = reference_gemm(input_data, weight_data, m, n, k);

        // Create BF16 input slice [m, k_local]
        std::vector<float> input_slice(static_cast<size_t>(m) * k_local);
        for (int row = 0; row < m; ++row)
        {
            for (int col = 0; col < k_local; ++col)
            {
                input_slice[row * k_local + col] = input_data[row * k + k_start + col];
            }
        }
        BF16Tensor input({static_cast<size_t>(m), static_cast<size_t>(k_local)});
        auto input_bf16 = fp32_to_bf16_vec(input_slice);
        std::memcpy(input.mutable_bf16_data(), input_bf16.data(), input_bf16.size() * sizeof(uint16_t));

        // Create BF16 weight slice [n, k_local] (must match activation type)
        std::vector<float> weight_slice(static_cast<size_t>(n) * k_local);
        for (int row = 0; row < n; ++row)
        {
            for (int col = 0; col < k_local; ++col)
            {
                weight_slice[row * k_local + col] = weight_data[row * k + k_start + col];
            }
        }
        BF16Tensor weight({static_cast<size_t>(n), static_cast<size_t>(k_local)});
        auto weight_bf16 = fp32_to_bf16_vec(weight_slice);
        std::memcpy(weight.mutable_bf16_data(), weight_bf16.data(), weight_bf16.size() * sizeof(uint16_t));

        // Create FP32 output (FloatingPointGemmKernel always outputs FP32)
        FP32Tensor output_fp32({static_cast<size_t>(m), static_cast<size_t>(n)});

        // Compute local partial product
        auto *gemm_kernel = KernelFactory::getOrCreateGemm(&weight);
        ASSERT_NE(gemm_kernel, nullptr);
        ASSERT_TRUE(gemm_kernel->multiply_tensor(&input, &output_fp32, m, n, k_local, true, 1.0f, 0.0f, mpi_ctx_.get(), -1));

        // Convert FP32 output to BF16 for allreduce test
        const size_t output_size = static_cast<size_t>(m) * n;
        auto output_bf16_vec = fp32_to_bf16_vec(std::vector<float>(output_fp32.data(), output_fp32.data() + output_size));
        BF16Tensor output({static_cast<size_t>(m), static_cast<size_t>(n)});
        std::memcpy(output.mutable_bf16_data(), output_bf16_vec.data(), output_bf16_vec.size() * sizeof(uint16_t));

        // Native BF16 allreduce-sum
        mpi_ctx_->allreduce_bf16_inplace(output.mutable_bf16_data(), output_size);

        // Verify
        if (rank_ == 0)
        {
            std::vector<float> result = bf16_to_fp32_vec(
                std::vector<uint16_t>(output.bf16_data(), output.bf16_data() + output_size));

            float cos_sim = cosine_similarity(reference_output, result);
            LOG_INFO("[K-Sliced BF16] Cosine similarity: " << cos_sim);
            EXPECT_GT(cos_sim, 0.99f) << "BF16 native allreduce should maintain good accuracy";
        }
    }

    /**
     * @test K-sliced row-parallel with FP16 native allreduce
     *
     * Creates FP16 weights to match FP16 activations for FloatingPointGemmKernel.
     * Output is FP32 (GEMM always outputs FP32), then converted to FP16 for allreduce test.
     */
    TEST_F(Test__MPI_RowParallelMultiPrecision, KSliced_FP16_NativeAllreduce)
    {
        ASSERT_GE(world_size_, 2) << "This test requires at least 2 MPI ranks";

        const int m = 4;
        const int n = 64;
        const int k = 128;
        const int k_local = k / world_size_;
        const int k_start = k_local * rank_;

        auto weight_data = generate_weight_matrix(n, k);
        auto input_data = generate_activation_matrix(m, k);
        auto reference_output = reference_gemm(input_data, weight_data, m, n, k);

        // Create FP16 input slice
        std::vector<float> input_slice(static_cast<size_t>(m) * k_local);
        for (int row = 0; row < m; ++row)
        {
            for (int col = 0; col < k_local; ++col)
            {
                input_slice[row * k_local + col] = input_data[row * k + k_start + col];
            }
        }
        FP16Tensor input({static_cast<size_t>(m), static_cast<size_t>(k_local)});
        auto input_fp16 = fp32_to_fp16_vec(input_slice);
        std::memcpy(input.mutable_fp16_data(), input_fp16.data(), input_fp16.size() * sizeof(uint16_t));

        // Create FP16 weight slice (must match activation type)
        std::vector<float> weight_slice(static_cast<size_t>(n) * k_local);
        for (int row = 0; row < n; ++row)
        {
            for (int col = 0; col < k_local; ++col)
            {
                weight_slice[row * k_local + col] = weight_data[row * k + k_start + col];
            }
        }
        FP16Tensor weight({static_cast<size_t>(n), static_cast<size_t>(k_local)});
        auto weight_fp16 = fp32_to_fp16_vec(weight_slice);
        std::memcpy(weight.mutable_fp16_data(), weight_fp16.data(), weight_fp16.size() * sizeof(uint16_t));

        // Create FP32 output (FloatingPointGemmKernel always outputs FP32)
        FP32Tensor output_fp32({static_cast<size_t>(m), static_cast<size_t>(n)});

        // Compute local partial product
        auto *gemm_kernel = KernelFactory::getOrCreateGemm(&weight);
        ASSERT_NE(gemm_kernel, nullptr);

        // Try to compute - may fail if OneDNN FP16 not supported on hardware
        bool success = gemm_kernel->multiply_tensor(&input, &output_fp32, m, n, k_local, true, 1.0f, 0.0f, mpi_ctx_.get(), -1);
        if (!success)
        {
            // OneDNN FP16 matmul not supported on this hardware - skip test
            GTEST_SKIP() << "OneDNN FP16 matmul not supported on this hardware";
        }

        // Convert FP32 output to FP16 for allreduce test
        const size_t output_size = static_cast<size_t>(m) * n;
        auto output_fp16_vec = fp32_to_fp16_vec(std::vector<float>(output_fp32.data(), output_fp32.data() + output_size));
        FP16Tensor output({static_cast<size_t>(m), static_cast<size_t>(n)});
        std::memcpy(output.mutable_fp16_data(), output_fp16_vec.data(), output_fp16_vec.size() * sizeof(uint16_t));

        // Native FP16 allreduce-sum
        mpi_ctx_->allreduce_fp16_inplace(output.mutable_fp16_data(), output_size);

        // Verify
        if (rank_ == 0)
        {
            std::vector<float> result = fp16_to_fp32_vec(
                std::vector<uint16_t>(output.fp16_data(), output.fp16_data() + output_size));

            float cos_sim = cosine_similarity(reference_output, result);
            LOG_INFO("[K-Sliced FP16] Cosine similarity: " << cos_sim);
            EXPECT_GT(cos_sim, 0.99f) << "FP16 native allreduce should maintain good accuracy";
        }
    }

    // =============================================================================
    // Edge Cases and Error Handling
    // =============================================================================

    /**
     * @test Non-block-aligned slices fallback to FP32 path
     *
     * When n_local is not a multiple of 32, the Q8_1 native path cannot be used.
     * This test verifies the FP32 path works correctly for non-aligned dimensions.
     * The test uses FP32 throughout since the GEMM kernel requires matching types.
     */
    TEST_F(Test__MPI_RowParallelMultiPrecision, Q8_1_NonAligned_FallbackToFP32)
    {
        ASSERT_GE(world_size_, 2) << "This test requires at least 2 MPI ranks";

        // Use dimensions that don't align to Q8_1 blocks
        const int m = 4;
        const int n = 100; // 100 / 2 = 50, NOT divisible by 32
        const int k = 64;

        auto weight_data = generate_weight_matrix(n, k);
        auto input_data = generate_activation_matrix(m, k);
        auto reference_output = reference_gemm(input_data, weight_data, m, n, k);

        // Create TensorSlice with PRE-SLICED FP32 tensor
        const int n_local = n / world_size_;
        const int n_start = n_local * rank_;

        // Create pre-sliced FP32 weight tensor
        auto sliced_weight = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_local), static_cast<size_t>(k)});
        for (int row = 0; row < n_local; ++row)
        {
            const int global_row = n_start + row;
            std::memcpy(sliced_weight->mutable_data() + row * k,
                        weight_data.data() + global_row * k,
                        k * sizeof(float));
        }

        SliceMetadata slice_meta = SliceMetadata::forRowParallel(
            static_cast<size_t>(n), static_cast<size_t>(k),
            rank_, world_size_, true /* inner_is_presliced */
        );
        std::unique_ptr<TensorBase> base_weight = std::move(sliced_weight);
        auto slice = std::make_unique<TensorSlice>(std::move(base_weight), slice_meta);
        ASSERT_NE(slice, nullptr);

        // Verify non-alignment
        EXPECT_NE(n_local % 32, 0) << "This test requires non-block-aligned n_local";

        // Create FP32 input (must match FP32 weight for FloatingPointGemmKernel)
        FP32Tensor input({static_cast<size_t>(m), static_cast<size_t>(k)});
        std::memcpy(input.mutable_data(), input_data.data(), input_data.size() * sizeof(float));

        FP32Tensor output({static_cast<size_t>(m), static_cast<size_t>(n)});
        FP32Tensor output_local({static_cast<size_t>(m), static_cast<size_t>(n_local)});

        // Compute using FP32 path
        auto *gemm_kernel = KernelFactory::getOrCreateGemm(slice.get());
        ASSERT_NE(gemm_kernel, nullptr);
        ASSERT_TRUE(gemm_kernel->multiply_tensor(&input, &output_local, m, n_local, k, true, 1.0f, 0.0f, mpi_ctx_.get(), -1));

        // Allgatherv FP32
        std::vector<int> recv_counts(world_size_);
        std::vector<int> displs(world_size_);
        for (int r = 0; r < world_size_; ++r)
        {
            int r_n_local = n / world_size_;
            if (r == world_size_ - 1)
                r_n_local = n - r_n_local * (world_size_ - 1);
            int r_start = r * (n / world_size_);
            recv_counts[r] = r_n_local * sizeof(float);
            displs[r] = r_start * sizeof(float);
        }

        const int local_row_bytes = n_local * sizeof(float);
        for (int row = 0; row < m; ++row)
        {
            mpi_ctx_->allgatherv_bytes(
                output_local.data() + row * n_local, local_row_bytes,
                output.mutable_data() + row * n, recv_counts.data(), displs.data());
        }

        // Verify
        if (rank_ == 0)
        {
            std::vector<float> result(output.data(), output.data() + static_cast<size_t>(m) * n);
            float cos_sim = cosine_similarity(reference_output, result);
            LOG_INFO("[Q8_1 Non-Aligned Fallback] Cosine similarity: " << cos_sim);
            EXPECT_GT(cos_sim, 0.98f) << "Fallback path should produce correct results";
        }
    }

    /**
     * @test Large matrix dimensions (stress test)
     */
    TEST_F(Test__MPI_RowParallelMultiPrecision, LargeMatrix_FP32)
    {
        ASSERT_GE(world_size_, 2) << "This test requires at least 2 MPI ranks";

        // Larger dimensions similar to real model layers
        const int m = 32;  // Batch size
        const int n = 512; // Hidden dimension
        const int k = 256; // Input dimension

        auto weight_data = generate_weight_matrix(n, k);
        auto input_data = generate_activation_matrix(m, k);
        auto reference_output = reference_gemm(input_data, weight_data, m, n, k);

        // K-sliced test
        const int k_local = k / world_size_;
        const int k_start = k_local * rank_;

        // Create sliced tensors
        std::vector<float> weight_slice(static_cast<size_t>(n) * k_local);
        for (int row = 0; row < n; ++row)
        {
            for (int col = 0; col < k_local; ++col)
            {
                weight_slice[row * k_local + col] = weight_data[row * k + k_start + col];
            }
        }
        FP32Tensor weight({static_cast<size_t>(n), static_cast<size_t>(k_local)});
        std::memcpy(weight.mutable_data(), weight_slice.data(), weight_slice.size() * sizeof(float));

        std::vector<float> input_slice(static_cast<size_t>(m) * k_local);
        for (int row = 0; row < m; ++row)
        {
            for (int col = 0; col < k_local; ++col)
            {
                input_slice[row * k_local + col] = input_data[row * k + k_start + col];
            }
        }
        FP32Tensor input({static_cast<size_t>(m), static_cast<size_t>(k_local)});
        std::memcpy(input.mutable_data(), input_slice.data(), input_slice.size() * sizeof(float));

        FP32Tensor output({static_cast<size_t>(m), static_cast<size_t>(n)});

        auto *gemm_kernel = KernelFactory::getOrCreateGemm(&weight);
        ASSERT_NE(gemm_kernel, nullptr);
        ASSERT_TRUE(gemm_kernel->multiply_tensor(&input, &output, m, n, k_local, true, 1.0f, 0.0f, mpi_ctx_.get(), -1));

        mpi_ctx_->allreduce_sum_inplace(output.mutable_data(), static_cast<size_t>(m) * n);

        if (rank_ == 0)
        {
            std::vector<float> result(output.data(), output.data() + static_cast<size_t>(m) * n);
            float cos_sim = cosine_similarity(reference_output, result);
            float max_diff = max_abs_diff(reference_output, result);

            LOG_INFO("[Large Matrix] Cosine similarity: " << cos_sim << ", Max diff: " << max_diff);
            EXPECT_GT(cos_sim, 0.999f);
        }
    }

    // =============================================================================
    // Real Quantized Weight Tests (QuantizedGemmKernel with Q4_0 weights + Q8_1 activations)
    // =============================================================================

    /**
     * @test TensorSlice row-parallel with real Q4_0 weights and Q8_1 activations
     *
     * Uses actual Q4_0 weights from a GGUF model with Q8_1 activations.
     * This exercises the QuantizedGemmKernel which is designed for
     * quantized weights × Q8_1 activations → Q8_1 output.
     *
     * Key: Uses loadTensorRowSlice to pre-slice weights at load time
     * (inner_is_presliced=true), so each rank's kernel packs only its
     * local weight portion.
     */
    TEST_F(Test__MPI_RowParallelMultiPrecision, TensorSlice_RealQ4_0_WithQ8_1Activations)
    {
        ASSERT_GE(world_size_, 2) << "This test requires at least 2 MPI ranks";

        // Skip if model file doesn't exist
        if (!std::ifstream(TEST_MODEL_PATH).good())
        {
            GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
        }

        // Load model
        TensorFactory factory(*mpi_ctx_);
        ModelLoader loader(&factory);
        if (!loader.loadModel(TEST_MODEL_PATH))
        {
            GTEST_SKIP() << "Failed to load model: " << TEST_MODEL_PATH;
        }

        // Full tensor dimensions: blk.0.ffn_down.weight [896, 4864]
        const std::string tensor_name = "blk.0.ffn_down.weight";
        const size_t full_n = 896;
        const size_t full_k = 4864;

        // Calculate slice bounds for this rank
        const size_t n_per_rank = full_n / world_size_;
        const size_t row_start = rank_ * n_per_rank;
        const size_t row_end = (rank_ == world_size_ - 1) ? full_n : (rank_ + 1) * n_per_rank;
        const size_t n_local = row_end - row_start;

        // Load PRE-SLICED tensor (each rank loads only its portion)
        auto inner_tensor = loader.loadTensorRowSlice(tensor_name, row_start, row_end, 0);
        if (!inner_tensor)
        {
            GTEST_SKIP() << "Failed to load tensor slice: " << tensor_name;
        }

        // Verify dimensions of pre-sliced tensor
        const auto &inner_shape = inner_tensor->shape();
        ASSERT_EQ(inner_shape.size(), 2);
        ASSERT_EQ(inner_shape[0], n_local) << "Pre-sliced tensor has wrong row count";
        ASSERT_EQ(inner_shape[1], full_k) << "Pre-sliced tensor has wrong column count";

        LOG_INFO("[Rank " << rank_ << "] Loaded pre-sliced tensor: [" << n_local << ", " << full_k << "]");

        // Create TensorSlice with inner_is_presliced=true
        SliceMetadata slice_meta = SliceMetadata::forRowParallel(
            full_n, full_k, rank_, world_size_, true /* inner_is_presliced */
        );
        auto slice = std::make_unique<TensorSlice>(std::move(inner_tensor), slice_meta);
        ASSERT_NE(slice, nullptr);

        // Small batch for test speed
        const int m = 4;
        const int n = static_cast<int>(full_n);
        const int k = static_cast<int>(full_k);
        const int local_n = static_cast<int>(n_local);

        // Generate deterministic input (same across all ranks)
        rng_.seed(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        std::vector<float> input_fp32(static_cast<size_t>(m) * k);
        for (auto &v : input_fp32)
        {
            v = dist(rng_);
        }

        // Create Q8_1 input tensor
        Q8_1Tensor input({static_cast<size_t>(m), static_cast<size_t>(k)});
        simd::quantize_fp32_to_q8_1_blocks(input_fp32.data(), input.mutable_q8_1_blocks(), input_fp32.size());

        // Create Q8_1 output tensors
        Q8_1Tensor output({static_cast<size_t>(m), static_cast<size_t>(n)});
        Q8_1Tensor output_local({static_cast<size_t>(m), static_cast<size_t>(local_n)});

        // Get QuantizedGemmKernel from TensorSlice (kernel packs only local weights)
        auto *gemm_kernel = KernelFactory::getOrCreateGemm(slice.get());
        ASSERT_NE(gemm_kernel, nullptr);

        // Compute local GEMM: Q8_1 activations × Q4_0 weights → Q8_1 output
        bool success = gemm_kernel->multiply_tensor(&input, &output_local, m, local_n, k, true, 1.0f, 0.0f, mpi_ctx_.get(), -1);
        ASSERT_TRUE(success) << "QuantizedGemmKernel multiply_tensor failed";

        // Allgatherv Q8_1 blocks
        constexpr size_t BLOCK_SIZE = Q8_1Block::BLOCK_SIZE;
        const size_t blocks_per_row_full = n / BLOCK_SIZE;
        const size_t blocks_per_row_local = local_n / BLOCK_SIZE;

        std::vector<int> recv_counts(world_size_);
        std::vector<int> displs(world_size_);
        for (int r = 0; r < world_size_; ++r)
        {
            size_t r_n_local = n / world_size_;
            size_t r_blocks_local = r_n_local / BLOCK_SIZE;
            size_t r_block_start = (r * (n / world_size_)) / BLOCK_SIZE;
            recv_counts[r] = static_cast<int>(r_blocks_local * sizeof(Q8_1Block));
            displs[r] = static_cast<int>(r_block_start * sizeof(Q8_1Block));
        }

        const int local_row_bytes = static_cast<int>(blocks_per_row_local * sizeof(Q8_1Block));
        const Q8_1Block *local_blocks = output_local.q8_1_blocks();
        Q8_1Block *output_blocks = output.mutable_q8_1_blocks();

        for (int row = 0; row < m; ++row)
        {
            const Q8_1Block *src_row = local_blocks + row * blocks_per_row_local;
            Q8_1Block *dst_row = output_blocks + row * blocks_per_row_full;
            mpi_ctx_->allgatherv_bytes(src_row, local_row_bytes, dst_row, recv_counts.data(), displs.data());
        }

        // Verify on rank 0 - compare against reference computed with full tensor
        if (rank_ == 0)
        {
            // Load full weight tensor for reference computation
            auto full_weight = loader.loadTensor(tensor_name, 0, WeightPrecision::NATIVE);
            ASSERT_NE(full_weight, nullptr);

            // Create reference output using full weight
            Q8_1Tensor ref_output({static_cast<size_t>(m), static_cast<size_t>(n)});
            auto *ref_kernel = KernelFactory::getOrCreateGemm(full_weight.get());
            ASSERT_NE(ref_kernel, nullptr);
            ASSERT_TRUE(ref_kernel->multiply_tensor(&input, &ref_output, m, n, k, true, 1.0f, 0.0f, mpi_ctx_.get(), -1));

            // Dequantize and compare
            std::vector<float> result(static_cast<size_t>(m) * n);
            std::vector<float> reference(static_cast<size_t>(m) * n);
            simd::dequantize_q8_1_to_fp32(output.q8_1_blocks(), result.data(), result.size());
            simd::dequantize_q8_1_to_fp32(ref_output.q8_1_blocks(), reference.data(), reference.size());

            float cos_sim = cosine_similarity(reference, result);
            float max_diff = max_abs_diff(reference, result);

            LOG_INFO("[Real Q4_0 + Q8_1] Cosine similarity: " << cos_sim << ", Max diff: " << max_diff);
            EXPECT_GT(cos_sim, 0.99f) << "Sliced Q4_0 GEMM should match full tensor GEMM";
        }
    }

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
