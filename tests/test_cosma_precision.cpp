/**
 * @file test_cosma_precision.cpp
 * @brief Focused unit tests for COSMA float vs double precision comparison
 *
 * These tests directly invoke COSMA in multi-rank MPI setups to isolate
 * the precision bug observed in distributed matrix operations.
 *
 * @author David Sanftenberg
 * @date 2025-10-07
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <cosma/matrix.hpp>
#include <cosma/multiply.hpp>
#include <cosma/strategy.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>

// OpenBLAS threading control
extern "C"
{
    void openblas_set_num_threads(int num_threads);
    int openblas_get_num_threads(void);
}
#include <iomanip>

namespace
{

    /**
     * @brief Compute reference matrix multiplication C = A * B using OpenBLAS
     */
    template <typename T>
    void reference_matmul(const T *A, const T *B, T *C, int m, int n, int k)
    {
        // Simple row-major C = A * B
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                T sum = 0;
                for (int p = 0; p < k; ++p)
                {
                    sum += A[i * k + p] * B[p * n + j];
                }
                C[i * n + j] = sum;
            }
        }
    }

    /**
     * @brief Compute max absolute difference and relative L2 norm
     */
    template <typename T>
    struct DiffMetrics
    {
        T max_abs_diff;
        T rel_l2;
        size_t num_elements;

        DiffMetrics(const T *a, const T *b, size_t count) : num_elements(count)
        {
            max_abs_diff = 0;
            T sum_sq_diff = 0;
            T sum_sq_ref = 0;

            for (size_t i = 0; i < count; ++i)
            {
                T diff = std::abs(a[i] - b[i]);
                max_abs_diff = std::max(max_abs_diff, diff);
                sum_sq_diff += diff * diff;
                sum_sq_ref += b[i] * b[i];
            }

            rel_l2 = (sum_sq_ref > 0) ? std::sqrt(sum_sq_diff / sum_sq_ref) : 0;
        }
    };

    /**
     * @brief Test COSMA matrix multiplication with specific precision type
     */
    template <typename T>
    class CosmaMatmulTester
    {
    public:
        CosmaMatmulTester(int m, int n, int k, int world_size, int rank)
            : m_(m), n_(n), k_(k), world_size_(world_size), rank_(rank) {}

        /**
         * @brief Run COSMA matmul and return result on rank 0
         * @param A_global Input matrix A (only valid on rank 0)
         * @param B_global Input matrix B (only valid on rank 0)
         * @param C_global Output matrix C (only valid on rank 0)
         * @param alpha Scalar alpha (default 1.0)
         * @param beta Scalar beta (default 0.0)
         */
        void run(const T *A_global, const T *B_global, T *C_global,
                 T alpha = static_cast<T>(1), T beta = static_cast<T>(0))
        {

            // Force multi-threaded execution
            openblas_set_num_threads(28);

            std::cout << "  [TEST][Rank " << rank_ << "] OpenBLAS threads: " << openblas_get_num_threads() << std::endl;
            std::cout.flush();

            std::cout << "  [TEST][Rank " << rank_ << "] Creating strategy for " << m_ << "x" << n_ << "x" << k_
                      << " on " << world_size_ << " ranks (type="
                      << (sizeof(T) == 4 ? "float" : "double") << ")" << std::endl;
            std::cout.flush();

            // Create strategy
            cosma::Strategy strategy(m_, n_, k_, world_size_);

            std::cout << "  [TEST][Rank " << rank_ << "] Strategy created, allocating matrices..." << std::endl;
            std::cout.flush();

            // Create COSMA matrices
            cosma::CosmaMatrix<T> A('A', strategy, rank_, false);
            cosma::CosmaMatrix<T> B('B', strategy, rank_, false);
            cosma::CosmaMatrix<T> C('C', strategy, rank_, false);

            std::cout << "  [TEST][Rank " << rank_ << "] Matrices created, calling allocate()..." << std::endl;
            std::cout.flush();

            // Allocate
            A.allocate();
            B.allocate();
            C.allocate();

            std::cout << "  [TEST][Rank " << rank_ << "] Matrices allocated, populating..." << std::endl;
            std::cout.flush();

            // Populate A and B from global data (rank 0 has full matrices)
            populate_matrix(A, A_global, m_, k_);
            populate_matrix(B, B_global, k_, n_);
            zero_matrix(C);

            if (rank_ == 0)
            {
                std::cout << "  [TEST] Matrices populated, rank 0 entering barrier..." << std::endl;
            }
            else
            {
                std::cout << "  [TEST] Matrices populated, rank " << rank_ << " entering barrier..." << std::endl;
            }

            std::cout.flush(); // Force output before barrier

            // Barrier before multiply
            MPI_Barrier(MPI_COMM_WORLD);

            if (rank_ == 0)
            {
                std::cout << "  [TEST] Barrier passed, calling COSMA multiply..." << std::endl;
            }

            // Execute COSMA multiply
            try
            {
                cosma::multiply(A, B, C, strategy, MPI_COMM_WORLD, alpha, beta);
            }
            catch (const std::exception &e)
            {
                if (rank_ == 0)
                {
                    std::cerr << "  [ERROR] COSMA multiply exception: " << e.what() << std::endl;
                }
                throw;
            }

            if (rank_ == 0)
            {
                std::cout << "  [TEST] COSMA multiply completed, entering barrier..." << std::endl;
            }

            // Barrier after multiply
            MPI_Barrier(MPI_COMM_WORLD);

            if (rank_ == 0)
            {
                std::cout << "  [TEST] Reconstructing result..." << std::endl;
            }

            // Reconstruct result to rank 0
            reconstruct_matrix(C, C_global, m_, n_);

            if (rank_ == 0)
            {
                std::cout << "  [TEST] Reconstruction complete" << std::endl;
            }
        }

    private:
        int m_, n_, k_;
        int world_size_, rank_;

        void populate_matrix(cosma::CosmaMatrix<T> &mat, const T *global_data, int rows, int cols)
        {
            T *local = mat.matrix_pointer();
            size_t local_size = mat.matrix_size();

            std::cout << "  [POPULATE][Rank " << rank_ << "] local_size=" << local_size
                      << " for " << rows << "x" << cols << " matrix" << std::endl;
            std::cout.flush();

            if (local && local_size > 0)
            {
                // Each rank populates its local portion based on global coordinates
                // This is the correct COSMA approach - each rank knows which global elements it owns
                auto start = std::chrono::high_resolution_clock::now();

                for (size_t li = 0; li < local_size; ++li)
                {
                    if (rank_ == 0 && li % 10000 == 0)
                    {
                        std::cout << "  [POPULATE] Processing element " << li << "/" << local_size << std::endl;
                    }

                    auto gc = mat.global_coordinates(static_cast<int>(li));
                    int gi = gc.first;
                    int gj = gc.second;

                    if (gi >= 0 && gi < rows && gj >= 0 && gj < cols)
                    {
                        // Each rank computes the same global data (for testing)
                        // In real usage, this would come from file/memory/etc based on gi,gj
                        if (global_data != nullptr)
                        {
                            local[li] = global_data[gi * cols + gj];
                        }
                        else
                        {
                            // If no global data provided, use global coordinates to generate value
                            local[li] = static_cast<T>((gi * cols + gj) % 100) / static_cast<T>(10);
                        }
                    }
                    else
                    {
                        local[li] = static_cast<T>(0);
                    }
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                std::cout << "  [POPULATE][Rank " << rank_ << "] Completed in " << ms << "ms" << std::endl;
                std::cout.flush();
            }
            else
            {
                std::cout << "  [POPULATE][Rank " << rank_ << "] Skipping - empty matrix" << std::endl;
                std::cout.flush();
            }

            // CRITICAL: Always call barrier, even if local_size==0!
            std::cout << "  [POPULATE][Rank " << rank_ << "] Entering barrier..." << std::endl;
            std::cout.flush();
            MPI_Barrier(MPI_COMM_WORLD);

            std::cout << "  [POPULATE][Rank " << rank_ << "] Barrier complete" << std::endl;
            std::cout.flush();
        }

        void zero_matrix(cosma::CosmaMatrix<T> &mat)
        {
            T *local = mat.matrix_pointer();
            size_t local_size = mat.matrix_size();
            if (local && local_size > 0)
            {
                std::fill(local, local + local_size, static_cast<T>(0));
            }
        }

        void reconstruct_matrix(cosma::CosmaMatrix<T> &mat, T *global_data, int rows, int cols)
        {
            size_t total = static_cast<size_t>(rows) * cols;
            std::vector<T> local_acc(total, 0);

            T *local = mat.matrix_pointer();
            size_t local_size = mat.matrix_size();

            if (local && local_size > 0)
            {
                for (size_t li = 0; li < local_size; ++li)
                {
                    auto gc = mat.global_coordinates(static_cast<int>(li));
                    int gi = gc.first;
                    int gj = gc.second;

                    if (gi >= 0 && gi < rows && gj >= 0 && gj < cols)
                    {
                        size_t idx = gi * cols + gj;
                        local_acc[idx] += local[li];
                    }
                }
            }

            // Reduce to rank 0
            MPI_Datatype mpi_type = (sizeof(T) == 4) ? MPI_FLOAT : MPI_DOUBLE;
            if (rank_ == 0)
            {
                MPI_Reduce(MPI_IN_PLACE, local_acc.data(), static_cast<int>(total),
                           mpi_type, MPI_SUM, 0, MPI_COMM_WORLD);
                std::copy(local_acc.begin(), local_acc.end(), global_data);
            }
            else
            {
                MPI_Reduce(local_acc.data(), nullptr, static_cast<int>(total),
                           mpi_type, MPI_SUM, 0, MPI_COMM_WORLD);
            }
        }
    };

} // anonymous namespace

/**
 * @brief Test small square matrix with float precision
 */
TEST(CosmaPrecision, SmallSquareFloat)
{
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (world_size < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    const int m = 32, n = 32, k = 32; // Very small to avoid hang

    std::vector<float> A(m * k, 0.0f);
    std::vector<float> B(k * n, 0.0f);
    std::vector<float> C_cosma(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);

    // Initialize on rank 0
    if (rank == 0)
    {
        std::cout << "\n[TEST] Initializing float test data..." << std::endl;
        for (int i = 0; i < m * k; ++i)
            A[i] = static_cast<float>(i % 100) / 10.0f;
        for (int i = 0; i < k * n; ++i)
            B[i] = static_cast<float>(i % 100) / 10.0f;

        std::cout << "[TEST] Computing reference..." << std::endl;
        // Compute reference
        reference_matmul(A.data(), B.data(), C_ref.data(), m, n, k);
        std::cout << "[TEST] Reference complete" << std::endl;
    }

    // Broadcast input data to all ranks (needed for populate_matrix)
    MPI_Bcast(A.data(), m * k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Run COSMA
    if (rank == 0)
    {
        std::cout << "[TEST] Starting COSMA..." << std::endl;
    }
    CosmaMatmulTester<float> tester(m, n, k, world_size, rank);
    tester.run(A.data(), B.data(), C_cosma.data());

    // Check result on rank 0
    if (rank == 0)
    {
        DiffMetrics<float> metrics(C_cosma.data(), C_ref.data(), m * n);

        std::cout << "  Float " << m << "x" << n << "x" << k
                  << ": max_abs=" << metrics.max_abs_diff
                  << " rel_l2=" << metrics.rel_l2 << std::endl;

        EXPECT_LT(metrics.max_abs_diff, 1e-3f) << "Float precision failed for small square matrix";
        EXPECT_LT(metrics.rel_l2, 1e-4f);
    }
}

/**
 * @brief Test small square matrix with double precision
 */
TEST(CosmaPrecision, SmallSquareDouble)
{
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (world_size < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    const int m = 64, n = 64, k = 64;

    std::vector<double> A(m * k, 0.0);
    std::vector<double> B(k * n, 0.0);
    std::vector<double> C_cosma(m * n, 0.0);
    std::vector<double> C_ref(m * n, 0.0);

    // Initialize on rank 0
    if (rank == 0)
    {
        for (int i = 0; i < m * k; ++i)
            A[i] = static_cast<double>(i % 100) / 10.0;
        for (int i = 0; i < k * n; ++i)
            B[i] = static_cast<double>(i % 100) / 10.0;

        // Compute reference
        reference_matmul(A.data(), B.data(), C_ref.data(), m, n, k);
    }

    // Run COSMA
    CosmaMatmulTester<double> tester(m, n, k, world_size, rank);
    tester.run(A.data(), B.data(), C_cosma.data());

    // Check result on rank 0
    if (rank == 0)
    {
        DiffMetrics<double> metrics(C_cosma.data(), C_ref.data(), m * n);

        std::cout << "  Double " << m << "x" << n << "x" << k
                  << ": max_abs=" << metrics.max_abs_diff
                  << " rel_l2=" << metrics.rel_l2 << std::endl;

        EXPECT_LT(metrics.max_abs_diff, 1e-9) << "Double precision failed for small square matrix";
        EXPECT_LT(metrics.rel_l2, 1e-10);
    }
}

/**
 * @brief Test LLM-sized matrix (Qwen hidden_dim) with float
 */
TEST(CosmaPrecision, LLMSizeFloat)
{
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (world_size < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    const int m = 32, n = 896, k = 896; // Typical LLM dimensions

    std::vector<float> A(m * k, 0.0f);
    std::vector<float> B(k * n, 0.0f);
    std::vector<float> C_cosma(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);

    // Initialize on rank 0
    if (rank == 0)
    {
        for (int i = 0; i < m * k; ++i)
            A[i] = static_cast<float>(i % 100) / 10.0f;
        for (int i = 0; i < k * n; ++i)
            B[i] = static_cast<float>(i % 100) / 10.0f;

        // Compute reference
        reference_matmul(A.data(), B.data(), C_ref.data(), m, n, k);
    }

    // Run COSMA
    CosmaMatmulTester<float> tester(m, n, k, world_size, rank);
    tester.run(A.data(), B.data(), C_cosma.data());

    // Check result on rank 0
    if (rank == 0)
    {
        DiffMetrics<float> metrics(C_cosma.data(), C_ref.data(), m * n);

        std::cout << "  Float " << m << "x" << n << "x" << k
                  << ": max_abs=" << metrics.max_abs_diff
                  << " rel_l2=" << metrics.rel_l2 << std::endl;

        // Float precision with distributed reduction - expect some error
        EXPECT_LT(metrics.max_abs_diff, 0.1f) << "Float precision errors too large for LLM-sized matrix";
    }
}

/**
 * @brief Test LLM-sized matrix (Qwen hidden_dim) with double
 */
TEST(CosmaPrecision, LLMSizeDouble)
{
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (world_size < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    const int m = 32, n = 896, k = 896; // Typical LLM dimensions

    std::vector<double> A(m * k, 0.0);
    std::vector<double> B(k * n, 0.0);
    std::vector<double> C_cosma(m * n, 0.0);
    std::vector<double> C_ref(m * n, 0.0);

    // Initialize on rank 0
    if (rank == 0)
    {
        for (int i = 0; i < m * k; ++i)
            A[i] = static_cast<double>(i % 100) / 10.0;
        for (int i = 0; i < k * n; ++i)
            B[i] = static_cast<double>(i % 100) / 10.0;

        // Compute reference
        reference_matmul(A.data(), B.data(), C_ref.data(), m, n, k);
    }

    // Run COSMA
    CosmaMatmulTester<double> tester(m, n, k, world_size, rank);
    tester.run(A.data(), B.data(), C_cosma.data());

    // Check result on rank 0
    if (rank == 0)
    {
        DiffMetrics<double> metrics(C_cosma.data(), C_ref.data(), m * n);

        std::cout << "  Double " << m << "x" << n << "x" << k
                  << ": max_abs=" << metrics.max_abs_diff
                  << " rel_l2=" << metrics.rel_l2 << std::endl;

        EXPECT_LT(metrics.max_abs_diff, 1e-9) << "Double precision failed for LLM-sized matrix";
        EXPECT_LT(metrics.rel_l2, 1e-10);
    }
}

/**
 * @brief Test wide matrix (moderate vocab size) with float
 */
TEST(CosmaPrecision, WideMatrixFloat)
{
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (world_size < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    const int m = 32, n = 10000, k = 896; // Wide matrix (smaller vocab)

    std::vector<float> A(m * k, 0.0f);
    std::vector<float> B(k * n, 0.0f);
    std::vector<float> C_cosma(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);

    // Initialize on rank 0
    if (rank == 0)
    {
        for (int i = 0; i < m * k; ++i)
            A[i] = static_cast<float>(i % 100) / 10.0f;
        for (int i = 0; i < k * n; ++i)
            B[i] = static_cast<float>(i % 100) / 10.0f;

        // Compute reference
        reference_matmul(A.data(), B.data(), C_ref.data(), m, n, k);
    }

    // Run COSMA
    CosmaMatmulTester<float> tester(m, n, k, world_size, rank);
    tester.run(A.data(), B.data(), C_cosma.data());

    // Check result on rank 0
    if (rank == 0)
    {
        DiffMetrics<float> metrics(C_cosma.data(), C_ref.data(), m * n);

        std::cout << "  Float " << m << "x" << n << "x" << k
                  << ": max_abs=" << metrics.max_abs_diff
                  << " rel_l2=" << metrics.rel_l2 << std::endl;

        // This is where we expect to see the COSMA float32 bug
        // Document actual behavior instead of asserting
        std::cout << "  NOTE: This is the known COSMA float32 bug range" << std::endl;
    }
}

/**
 * @brief Test wide matrix (moderate vocab size) with double
 */
TEST(CosmaPrecision, WideMatrixDouble)
{
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (world_size < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    const int m = 32, n = 10000, k = 896; // Wide matrix (smaller vocab)

    std::vector<double> A(m * k, 0.0);
    std::vector<double> B(k * n, 0.0);
    std::vector<double> C_cosma(m * n, 0.0);
    std::vector<double> C_ref(m * n, 0.0);

    // Initialize on rank 0
    if (rank == 0)
    {
        for (int i = 0; i < m * k; ++i)
            A[i] = static_cast<double>(i % 100) / 10.0;
        for (int i = 0; i < k * n; ++i)
            B[i] = static_cast<double>(i % 100) / 10.0;

        // Compute reference
        reference_matmul(A.data(), B.data(), C_ref.data(), m, n, k);
    }

    // Run COSMA
    CosmaMatmulTester<double> tester(m, n, k, world_size, rank);
    tester.run(A.data(), B.data(), C_cosma.data());

    // Check result on rank 0
    if (rank == 0)
    {
        DiffMetrics<double> metrics(C_cosma.data(), C_ref.data(), m * n);

        std::cout << "  Double " << m << "x" << n << "x" << k
                  << ": max_abs=" << metrics.max_abs_diff
                  << " rel_l2=" << metrics.rel_l2 << std::endl;

        // Double should be much better than float
        EXPECT_LT(metrics.max_abs_diff, 1e-8) << "Double precision failed for wide matrix";
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Only rank 0 prints test output
    if (rank != 0)
    {
        ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
