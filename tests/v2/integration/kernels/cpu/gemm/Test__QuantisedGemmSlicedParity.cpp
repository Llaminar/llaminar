/**
 * @file Test__QuantisedGemmSlicedParity.cpp
 * @brief Integration test verifying sliced GEMM across MPI ranks produces same result as unsliced GEMM
 * @author David Sanftenberg
 *
 * This test verifies that when two MPI ranks each perform GEMM on their row-slice
 * of a weight tensor, and the results are concatenated, the output matches
 * a single unsliced GEMM operation.
 *
 * Key verification:
 * - Rank 0 loads rows [0, N/2) and does GEMM → partial_result_0
 * - Rank 1 loads rows [N/2, N) and does GEMM → partial_result_1
 * - Concatenate(partial_result_0, partial_result_1) == full_gemm_result
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "../../src/v2/loaders/ModelLoader.h"
#include "../../src/v2/loaders/WeightManager.h"
#include "../../src/v2/tensors/TensorSlice.h"
#include "../../src/v2/tensors/TensorFactory.h"
#include "../../src/v2/kernels/KernelFactory.h"
#include "../../src/v2/utils/MPIContext.h"
#include "../../src/v2/utils/Logger.h"
#include "../../src/v2/backends/DeviceId.h"

namespace llaminar2
{
    namespace test
    {

        class QuantisedGemmSlicedParityTest : public ::testing::Test
        {
        protected:
            static constexpr const char *MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

            int rank_ = 0;
            int world_size_ = 1;

            void SetUp() override
            {
                MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
                MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

                // This test requires exactly 2 MPI ranks
                ASSERT_EQ(world_size_, 2) << "This test requires exactly 2 MPI ranks";
            }

            void TearDown() override
            {
                MPI_Barrier(MPI_COMM_WORLD);
            }

            /**
             * @brief Helper: run GEMM via multiply_tensor, writing result into a raw float vector.
             */
            bool multiplyToVector(ITensorGemm *gemm, const float *A_data, float *C_data,
                                  int M, int N, int K)
            {
                FP32Tensor A_tensor(std::vector<size_t>{(size_t)M, (size_t)K});
                std::memcpy(A_tensor.mutable_data(), A_data, (size_t)M * K * sizeof(float));
                FP32Tensor C_tensor(std::vector<size_t>{(size_t)M, (size_t)N});
                bool ok = gemm->multiply_tensor(&A_tensor, &C_tensor, M, N, K);
                if (ok)
                    std::memcpy(C_data, C_tensor.data(), (size_t)M * N * sizeof(float));
                return ok;
            }

            /**
             * @brief Compute max absolute difference between two vectors
             */
            float maxAbsDiff(const std::vector<float> &a, const std::vector<float> &b)
            {
                EXPECT_EQ(a.size(), b.size());
                float max_diff = 0.0f;
                for (size_t i = 0; i < a.size(); ++i)
                {
                    max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
                }
                return max_diff;
            }

            /**
             * @brief Compute mean absolute difference between two vectors
             */
            float meanAbsDiff(const std::vector<float> &a, const std::vector<float> &b)
            {
                EXPECT_EQ(a.size(), b.size());
                float sum = 0.0f;
                for (size_t i = 0; i < a.size(); ++i)
                {
                    sum += std::abs(a[i] - b[i]);
                }
                return sum / a.size();
            }
        };

        /**
         * @brief Test that sliced GEMM across 2 ranks matches unsliced GEMM
         *
         * For row-parallel GEMM with weight matrix W[N, K]:
         * - Full GEMM: C = A * W^T where A[M, K], W[N, K], C[M, N]
         * - Sliced: W split into W0[N/2, K] and W1[N/2, K]
         *   - C0 = A * W0^T → partial [M, N/2] result for rows 0..N/2
         *   - C1 = A * W1^T → partial [M, N/2] result for rows N/2..N
         *   - Concatenate C0 and C1 → full [M, N] result
         */
        TEST_F(QuantisedGemmSlicedParityTest, SlicedGemmMatchesUnslicedGemm)
        {
            // Each rank creates its own loader and context
            auto mpi_ctx = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
            TensorFactory factory(*mpi_ctx);
            ModelLoader loader(&factory);

            if (!loader.loadModel(MODEL_PATH))
            {
                GTEST_SKIP() << "Model file not found: " << MODEL_PATH;
            }

            // Use a row-parallel weight: attn_output (Wo)
            // Shape: [896, 896] (output_dim, input_dim)
            const std::string tensor_name = "blk.0.attn_output.weight";

            // Get tensor dimensions
            const size_t full_rows = 896;
            const size_t full_cols = 896;

            // Calculate row slice bounds for each rank
            size_t rows_per_rank = full_rows / world_size_;
            size_t row_start = rank_ * rows_per_rank;
            size_t row_end = (rank_ == world_size_ - 1) ? full_rows : (rank_ + 1) * rows_per_rank;
            size_t slice_rows = row_end - row_start;

            LOG_DEBUG("Rank " << rank_ << " row slice: [" << row_start << ", " << row_end << ") = " << slice_rows << " rows");

            // Load only the row slice for this rank (memory-efficient)
            auto slice_tensor = loader.loadTensorRowSlice(tensor_name, row_start, row_end, DeviceId::cpu());
            ASSERT_NE(slice_tensor, nullptr) << "Failed to load tensor slice";
            ASSERT_EQ(slice_tensor->shape()[0], slice_rows);
            ASSERT_EQ(slice_tensor->shape()[1], full_cols);

            // Create test input: A[M, K] where M=4 (batch), K=896 (input_dim)
            const int M = 4;
            const int K = static_cast<int>(full_cols);
            const int N = static_cast<int>(full_rows);
            const int slice_N = static_cast<int>(slice_rows);

            // Create deterministic input data (same on all ranks)
            std::vector<float> input_A(M * K);
            for (int i = 0; i < M * K; ++i)
            {
                // Use deterministic pattern: sin-based to avoid all-zeros or simple patterns
                input_A[i] = std::sin(static_cast<float>(i) * 0.01f) * 0.5f;
            }

            // ============================================================
            // Part 1: Compute sliced GEMM on each rank
            // ============================================================

            // Create GEMM kernel for the slice
            auto slice_gemm = slice_tensor->createGemm();
            ASSERT_NE(slice_gemm, nullptr) << "Failed to create GEMM kernel for slice";

            // Output buffer for slice: [M, slice_N]
            std::vector<float> slice_output(M * slice_N, 0.0f);

            // Perform GEMM: slice_output = input_A * slice_tensor^T
            // [M, slice_N] = [M, K] * [slice_N, K]^T
            bool success = multiplyToVector(
                slice_gemm.get(),
                input_A.data(),
                slice_output.data(),
                M,       // rows of A
                slice_N, // cols of output (rows of B)
                K        // cols of A (cols of B)
            );
            ASSERT_TRUE(success) << "Sliced GEMM failed on rank " << rank_;

            LOG_DEBUG("Rank " << rank_ << " sliced GEMM complete, output size: " << slice_output.size());

            // ============================================================
            // Part 2: Gather all slices to reconstruct full output
            // ============================================================

            // Full output buffer: [M, N]
            std::vector<float> gathered_output(M * N, 0.0f);

            // Use MPI_Gather to collect all slices
            std::vector<float> all_slices;
            if (rank_ == 0)
            {
                all_slices.resize(M * N);
            }

            MPI_Gather(
                slice_output.data(),
                M * slice_N,
                MPI_FLOAT,
                all_slices.data(),
                M * slice_N,
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);

            // On rank 0, reconstruct the full output by placing slices in correct positions
            if (rank_ == 0)
            {
                // all_slices layout: [rank0_slice, rank1_slice] where each is [M, slice_N]
                // We need to interleave columns: output[m, n] where n determines which rank
                for (int m = 0; m < M; ++m)
                {
                    for (int r = 0; r < world_size_; ++r)
                    {
                        size_t r_row_start = r * rows_per_rank;
                        size_t r_slice_n = (r == world_size_ - 1) ? (N - r_row_start) : rows_per_rank;

                        for (size_t local_n = 0; local_n < r_slice_n; ++local_n)
                        {
                            size_t global_n = r_row_start + local_n;
                            // Source: all_slices[r][m, local_n] = all_slices[r * M * slice_N + m * slice_N + local_n]
                            // Dest: gathered_output[m, global_n] = gathered_output[m * N + global_n]
                            gathered_output[m * N + global_n] = all_slices[r * M * static_cast<int>(rows_per_rank) + m * static_cast<int>(rows_per_rank) + local_n];
                        }
                    }
                }
            }

            // ============================================================
            // Part 3: Compute full unsliced GEMM on rank 0 for reference
            // ============================================================

            std::vector<float> full_output(M * N, 0.0f);

            if (rank_ == 0)
            {
                // Load the full tensor for reference
                auto full_tensor = loader.loadTensor(tensor_name, DeviceId::cpu());
                ASSERT_NE(full_tensor, nullptr) << "Failed to load full tensor";

                auto full_gemm = full_tensor->createGemm();
                ASSERT_NE(full_gemm, nullptr) << "Failed to create GEMM kernel for full tensor";

                success = multiplyToVector(full_gemm.get(), input_A.data(), full_output.data(), M, N, K);
                ASSERT_TRUE(success) << "Full GEMM failed";

                LOG_DEBUG("Full GEMM complete, output size: " << full_output.size());
            }

            // ============================================================
            // Part 4: Compare results on rank 0
            // ============================================================

            if (rank_ == 0)
            {
                float max_diff = maxAbsDiff(gathered_output, full_output);
                float mean_diff = meanAbsDiff(gathered_output, full_output);

                LOG_INFO("Sliced vs Full GEMM comparison:");
                LOG_INFO("  Max absolute difference: " << max_diff);
                LOG_INFO("  Mean absolute difference: " << mean_diff);

                // The results should be bitwise identical since we're using the same
                // quantized weights, just loaded in slices
                EXPECT_LT(max_diff, 1e-5f) << "Sliced GEMM results differ from full GEMM";
                EXPECT_LT(mean_diff, 1e-6f) << "Mean difference too high";

                // Verify some specific values for debugging
                LOG_DEBUG("Sample values comparison:");
                for (int i = 0; i < std::min(5, M * N); ++i)
                {
                    LOG_DEBUG("  [" << i << "] gathered=" << gathered_output[i] << " full=" << full_output[i]
                                    << " diff=" << std::abs(gathered_output[i] - full_output[i]));
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        /**
         * @brief Test sliced GEMM with TensorSlice wrapper (as used by WeightManager)
         */
        TEST_F(QuantisedGemmSlicedParityTest, TensorSliceGemmMatchesUnslicedGemm)
        {
            auto mpi_ctx = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
            TensorFactory factory(*mpi_ctx);
            ModelLoader loader(&factory);

            if (!loader.loadModel(MODEL_PATH))
            {
                GTEST_SKIP() << "Model file not found: " << MODEL_PATH;
            }

            const std::string tensor_name = "blk.0.ffn_down.weight"; // [896, 4864]

            const size_t full_rows = 896;
            const size_t full_cols = 4864;

            // Calculate slice bounds
            size_t rows_per_rank = full_rows / world_size_;
            size_t row_start = rank_ * rows_per_rank;
            size_t row_end = (rank_ == world_size_ - 1) ? full_rows : (rank_ + 1) * rows_per_rank;
            size_t slice_rows = row_end - row_start;

            // Load pre-sliced tensor and wrap in TensorSlice
            auto inner_tensor = loader.loadTensorRowSlice(tensor_name, row_start, row_end, DeviceId::cpu());
            ASSERT_NE(inner_tensor, nullptr);

            // Create TensorSlice with inner_is_presliced=true
            // forRowParallel signature: (original_rows, original_cols, rank, world_size, inner_is_presliced)
            SliceMetadata meta = SliceMetadata::forRowParallel(
                full_rows, full_cols, rank_, world_size_, true /* inner_is_presliced */
            );

            auto tensor_slice = std::make_unique<TensorSlice>(std::move(inner_tensor), meta);

            // Test dimensions
            const int M = 8;
            const int K = static_cast<int>(full_cols);
            const int N = static_cast<int>(full_rows);
            const int slice_N = static_cast<int>(slice_rows);

            // Deterministic input
            std::vector<float> input_A(M * K);
            for (int i = 0; i < M * K; ++i)
            {
                input_A[i] = std::cos(static_cast<float>(i) * 0.007f) * 0.3f;
            }

            // Create GEMM from TensorSlice
            auto slice_gemm = tensor_slice->createGemm();
            ASSERT_NE(slice_gemm, nullptr);

            // Compute sliced GEMM
            std::vector<float> slice_output(M * slice_N, 0.0f);
            bool success = multiplyToVector(slice_gemm.get(), input_A.data(), slice_output.data(), M, slice_N, K);
            ASSERT_TRUE(success);

            // Gather results
            std::vector<float> gathered_output(M * N, 0.0f);
            std::vector<float> all_slices;
            if (rank_ == 0)
            {
                all_slices.resize(M * N);
            }

            MPI_Gather(
                slice_output.data(), M * slice_N, MPI_FLOAT,
                all_slices.data(), M * slice_N, MPI_FLOAT,
                0, MPI_COMM_WORLD);

            if (rank_ == 0)
            {
                for (int m = 0; m < M; ++m)
                {
                    for (int r = 0; r < world_size_; ++r)
                    {
                        size_t r_row_start = r * rows_per_rank;
                        for (size_t local_n = 0; local_n < rows_per_rank; ++local_n)
                        {
                            size_t global_n = r_row_start + local_n;
                            gathered_output[m * N + global_n] = all_slices[r * M * static_cast<int>(rows_per_rank) + m * static_cast<int>(rows_per_rank) + local_n];
                        }
                    }
                }
            }

            // Compute full GEMM on rank 0
            std::vector<float> full_output(M * N, 0.0f);
            if (rank_ == 0)
            {
                auto full_tensor = loader.loadTensor(tensor_name, DeviceId::cpu());
                ASSERT_NE(full_tensor, nullptr);
                auto full_gemm = full_tensor->createGemm();
                ASSERT_NE(full_gemm, nullptr);
                success = multiplyToVector(full_gemm.get(), input_A.data(), full_output.data(), M, N, K);
                ASSERT_TRUE(success);
            }

            // Compare
            if (rank_ == 0)
            {
                float max_diff = maxAbsDiff(gathered_output, full_output);
                float mean_diff = meanAbsDiff(gathered_output, full_output);

                LOG_INFO("TensorSlice GEMM comparison (ffn_down):");
                LOG_INFO("  Max absolute difference: " << max_diff);
                LOG_INFO("  Mean absolute difference: " << mean_diff);

                EXPECT_LT(max_diff, 1e-5f);
                EXPECT_LT(mean_diff, 1e-6f);
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        /**
         * @brief Test with larger batch size to stress test the slicing
         */
        TEST_F(QuantisedGemmSlicedParityTest, LargeBatchSlicedGemm)
        {
            auto mpi_ctx = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
            TensorFactory factory(*mpi_ctx);
            ModelLoader loader(&factory);

            if (!loader.loadModel(MODEL_PATH))
            {
                GTEST_SKIP() << "Model file not found: " << MODEL_PATH;
            }

            const std::string tensor_name = "blk.0.attn_output.weight";

            const size_t full_rows = 896;
            const size_t full_cols = 896;

            size_t rows_per_rank = full_rows / world_size_;
            size_t row_start = rank_ * rows_per_rank;
            size_t row_end = (rank_ == world_size_ - 1) ? full_rows : (rank_ + 1) * rows_per_rank;
            size_t slice_rows = row_end - row_start;

            auto slice_tensor = loader.loadTensorRowSlice(tensor_name, row_start, row_end, DeviceId::cpu());
            ASSERT_NE(slice_tensor, nullptr);

            // Large batch
            const int M = 64;
            const int K = static_cast<int>(full_cols);
            const int N = static_cast<int>(full_rows);
            const int slice_N = static_cast<int>(slice_rows);

            std::vector<float> input_A(M * K);
            for (int i = 0; i < M * K; ++i)
            {
                input_A[i] = std::sin(static_cast<float>(i) * 0.003f + 0.5f) * 0.4f;
            }

            auto slice_gemm = slice_tensor->createGemm();
            ASSERT_NE(slice_gemm, nullptr);

            std::vector<float> slice_output(M * slice_N, 0.0f);
            bool success = multiplyToVector(slice_gemm.get(), input_A.data(), slice_output.data(), M, slice_N, K);
            ASSERT_TRUE(success);

            std::vector<float> gathered_output(M * N, 0.0f);
            std::vector<float> all_slices;
            if (rank_ == 0)
            {
                all_slices.resize(M * N);
            }

            MPI_Gather(
                slice_output.data(), M * slice_N, MPI_FLOAT,
                all_slices.data(), M * slice_N, MPI_FLOAT,
                0, MPI_COMM_WORLD);

            if (rank_ == 0)
            {
                for (int m = 0; m < M; ++m)
                {
                    for (int r = 0; r < world_size_; ++r)
                    {
                        size_t r_row_start = r * rows_per_rank;
                        for (size_t local_n = 0; local_n < rows_per_rank; ++local_n)
                        {
                            size_t global_n = r_row_start + local_n;
                            gathered_output[m * N + global_n] = all_slices[r * M * static_cast<int>(rows_per_rank) + m * static_cast<int>(rows_per_rank) + local_n];
                        }
                    }
                }
            }

            std::vector<float> full_output(M * N, 0.0f);
            if (rank_ == 0)
            {
                auto full_tensor = loader.loadTensor(tensor_name, DeviceId::cpu());
                ASSERT_NE(full_tensor, nullptr);
                auto full_gemm = full_tensor->createGemm();
                ASSERT_NE(full_gemm, nullptr);
                success = multiplyToVector(full_gemm.get(), input_A.data(), full_output.data(), M, N, K);
                ASSERT_TRUE(success);
            }

            if (rank_ == 0)
            {
                float max_diff = maxAbsDiff(gathered_output, full_output);
                float mean_diff = meanAbsDiff(gathered_output, full_output);

                LOG_INFO("Large batch (M=64) sliced GEMM comparison:");
                LOG_INFO("  Max absolute difference: " << max_diff);
                LOG_INFO("  Mean absolute difference: " << mean_diff);

                EXPECT_LT(max_diff, 1e-5f);
                EXPECT_LT(mean_diff, 1e-6f);
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

    } // namespace test
} // namespace llaminar2

// Main with MPI initialization
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
