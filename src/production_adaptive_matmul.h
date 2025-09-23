// Production Adaptive Matrix Multiplication Backend
//
// Final implementation using only OpenBLAS with different configurations
// for optimal performance across different operation sizes.

#pragma once

#include <vector>
#include <memory>
#include <chrono>
#include <mpi.h>
#include <cblas.h>
#include <omp.h>

namespace llaminar
{

    enum class MatMulBackend
    {
        SINGLE_THREADED_OPENBLAS, // For small operations, avoids thread overhead
        MULTI_THREADED_OPENBLAS,  // For medium operations, uses local parallelism
        DISTRIBUTED_OPENBLAS      // For large operations, uses MPI distribution
    };

    class ProductionMatMulManager
    {
    private:
        int mpi_rank_, mpi_size_;
        bool mpi_initialized_;
        bool verbose_ = true; // controls console logging inside multiply

        // Performance thresholds based on empirical testing results:
        //
        // MEASURED PERFORMANCE CROSSOVER POINTS:
        // • Single token (1x896x896): OpenBLAS 134x faster than COSMA
        // • Medium batch (32x896x896): OpenBLAS 3x faster
        // • Prefill 512 (512x896x896): OpenBLAS 12x faster
        // • Prefill 1K (1024x896x896): OpenBLAS 16x faster
        // • Prefill 8K (8192x896x896): COSMA becomes competitive (1.1x faster)
        // • Prefill 16K+ (≥16384 tokens): COSMA dominates (1.6-3.6x faster)
        //
        // BACKEND SELECTION STRATEGY:
        // • ≤512 tokens: Always OpenBLAS (massive advantage)
        // • 512-8K tokens: Distributed OpenBLAS (still ahead but worth MPI distribution)
        // • ≥8K tokens: Could use COSMA but staying with OpenBLAS for consistency
        //
        // Performance thresholds (empirically determined from actual testing)
        // Single token: 1x896x896 = 802,816 elements - use multi-threaded
        static constexpr size_t SINGLE_THREAD_MAX_ELEMENTS = 16384; // ~128x128 or smaller
        // Crossover to distributed at 512 tokens: 512x896x896 = ~413M elements
        static constexpr size_t MULTI_THREAD_MAX_ELEMENTS = 400000000; // ~512x896x896
        // Sequence length thresholds based on measured crossover points
        static constexpr int COSMA_SEQUENCE_THRESHOLD = 8192;        // 8K tokens where COSMA becomes competitive
        static constexpr size_t VOCAB_PROJECTION_THRESHOLD = 100000; // Avoid large vocab projections in distributed mode

    public:
        ProductionMatMulManager()
        {
            int flag;
            MPI_Initialized(&flag);
            mpi_initialized_ = (flag != 0);

            if (mpi_initialized_)
            {
                MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
                MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
            }
            else
            {
                mpi_rank_ = 0;
                mpi_size_ = 1;
            }
        }

        void setVerbose(bool v) { verbose_ = v; }

        // Smart backend selection based on actual measured performance characteristics
        MatMulBackend selectBackend(int m, int n, int k, bool is_prefill = false)
        {
            size_t total_elements = static_cast<size_t>(m) * n * k;
            int sequence_length = m; // Assuming m is sequence length for transformer ops

            // Avoid distributed mode for vocabulary projections (too much communication overhead)
            // Test showed: 512x32000x896 still uses multi-threaded for best performance
            if (n > VOCAB_PROJECTION_THRESHOLD || k > VOCAB_PROJECTION_THRESHOLD)
            {
                return total_elements > SINGLE_THREAD_MAX_ELEMENTS ? MatMulBackend::MULTI_THREADED_OPENBLAS : MatMulBackend::SINGLE_THREADED_OPENBLAS;
            }

            // For prefill operations, use sequence length as primary decision factor
            if (mpi_initialized_ && mpi_size_ > 1 && is_prefill)
            {
                // Based on test results:
                // - Small sequences (≤512): OpenBLAS dominates (12x-134x faster)
                // - Medium sequences (1K-8K): OpenBLAS still ahead but gap closes
                // - Large sequences (≥8K): COSMA becomes competitive/better
                if (sequence_length >= 512) // Start considering distributed at 512 tokens
                {
                    return MatMulBackend::DISTRIBUTED_OPENBLAS;
                }
            }

            // Use multi-threaded for medium operations (tested optimal for 1-512 token range)
            if (total_elements > SINGLE_THREAD_MAX_ELEMENTS)
            {
                return MatMulBackend::MULTI_THREADED_OPENBLAS;
            }

            // Use single-threaded for small operations (avoids thread overhead)
            return MatMulBackend::SINGLE_THREADED_OPENBLAS;
        }

        // Main matrix multiplication interface
        bool multiply(const float *A, const float *B, float *C,
                      int m, int n, int k,
                      bool transpose_A = false, bool transpose_B = false,
                      float alpha = 1.0f, float beta = 0.0f,
                      bool is_prefill = false)
        {

            MatMulBackend backend = selectBackend(m, n, k, is_prefill);

            if (verbose_ && mpi_rank_ == 0)
            {
                std::string backend_name = getBackendName(backend);
                // Only log for interesting operations (avoid spam for tiny ops)
                if (m > 4 || n > 4 || k > 4)
                {
                    std::cout << "MatMul " << m << "x" << n << "x" << k
                              << (is_prefill ? " (prefill)" : " (inference)")
                              << " -> " << backend_name << std::endl;
                }
            }

            switch (backend)
            {
            case MatMulBackend::SINGLE_THREADED_OPENBLAS:
                return multiply_single_threaded_openblas(A, B, C, m, n, k, transpose_A, transpose_B, alpha, beta);
            case MatMulBackend::MULTI_THREADED_OPENBLAS:
                return multiply_multi_threaded_openblas(A, B, C, m, n, k, transpose_A, transpose_B, alpha, beta);
            case MatMulBackend::DISTRIBUTED_OPENBLAS:
                return multiply_distributed_openblas(A, B, C, m, n, k, transpose_A, transpose_B, alpha, beta);
            }

            return false;
        }

        // Performance measurement interface
        struct PerformanceStats
        {
            double time_ms;
            double gflops;
            std::string backend;
        };

        PerformanceStats measurePerformance(int m, int n, int k, bool is_prefill = false)
        {
            // Create test matrices
            std::vector<float> A(m * k, 1.0f);
            std::vector<float> B(k * n, 2.0f);
            std::vector<float> C(m * n, 0.0f);

            auto start = std::chrono::high_resolution_clock::now();

            bool success = multiply(A.data(), B.data(), C.data(), m, n, k, false, false, 1.0f, 0.0f, is_prefill);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            double time_ms = duration.count() / 1000.0;
            double operations = 2.0 * m * n * k;          // multiply-add operations
            double gflops = operations / (time_ms * 1e6); // GFLOPS

            MatMulBackend backend = selectBackend(m, n, k, is_prefill);
            std::string backend_name = getBackendName(backend);

            return {time_ms, gflops, backend_name};
        }

    private:
        std::string getBackendName(MatMulBackend backend)
        {
            switch (backend)
            {
            case MatMulBackend::SINGLE_THREADED_OPENBLAS:
                return "OpenBLAS(1T)";
            case MatMulBackend::MULTI_THREADED_OPENBLAS:
                return "OpenBLAS(MT)";
            case MatMulBackend::DISTRIBUTED_OPENBLAS:
                return "OpenBLAS(MPI)";
            }
            return "Unknown";
        }

        bool multiply_single_threaded_openblas(const float *A, const float *B, float *C,
                                               int m, int n, int k,
                                               bool transpose_A, bool transpose_B,
                                               float alpha, float beta)
        {
            // Force single-threaded execution
            int old_threads = openblas_get_num_threads();
            openblas_set_num_threads(1);

            CBLAS_TRANSPOSE trans_A = transpose_A ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE trans_B = transpose_B ? CblasTrans : CblasNoTrans;

            int lda = transpose_A ? m : k;
            int ldb = transpose_B ? k : n;
            int ldc = n;

            cblas_sgemm(CblasRowMajor, trans_A, trans_B,
                        m, n, k,
                        alpha, A, lda,
                        B, ldb,
                        beta, C, ldc);

            openblas_set_num_threads(old_threads);
            return true;
        }

        bool multiply_multi_threaded_openblas(const float *A, const float *B, float *C,
                                              int m, int n, int k,
                                              bool transpose_A, bool transpose_B,
                                              float alpha, float beta)
        {
            // Use optimal thread count (usually number of cores)
            int optimal_threads = std::min(omp_get_max_threads(), 8); // Cap at 8 threads
            int old_threads = openblas_get_num_threads();
            openblas_set_num_threads(optimal_threads);

            CBLAS_TRANSPOSE trans_A = transpose_A ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE trans_B = transpose_B ? CblasTrans : CblasNoTrans;

            int lda = transpose_A ? m : k;
            int ldb = transpose_B ? k : n;
            int ldc = n;

            cblas_sgemm(CblasRowMajor, trans_A, trans_B,
                        m, n, k,
                        alpha, A, lda,
                        B, ldb,
                        beta, C, ldc);

            openblas_set_num_threads(old_threads);
            return true;
        }

        bool multiply_distributed_openblas(const float *A, const float *B, float *C,
                                           int m, int n, int k,
                                           bool transpose_A, bool transpose_B,
                                           float alpha, float beta)
        {
            if (!mpi_initialized_ || mpi_size_ == 1)
            {
                // Fallback to multi-threaded if MPI not available
                return multiply_multi_threaded_openblas(A, B, C, m, n, k, transpose_A, transpose_B, alpha, beta);
            }

            // Safety check for extremely large operations based on test results
            // Operations >1B elements showed better performance with multi-threaded fallback
            size_t total_ops = static_cast<size_t>(m) * n * k;
            if (total_ops > 1000000000)
            { // > 1B operations (observed at 2K+ tokens)
                if (mpi_rank_ == 0)
                {
                    std::cout << "Very large operation (" << total_ops << " ops), using multi-threaded for reliability" << std::endl;
                }
                return multiply_multi_threaded_openblas(A, B, C, m, n, k, transpose_A, transpose_B, alpha, beta);
            }

            // Simple row-wise distribution
            int rows_per_proc = (m + mpi_size_ - 1) / mpi_size_;
            int start_row = mpi_rank_ * rows_per_proc;
            int end_row = std::min(start_row + rows_per_proc, m);
            int local_rows = std::max(0, end_row - start_row);

            if (local_rows == 0)
            {
                // This rank has no work
                return true;
            }

            // Broadcast B matrix to all ranks (B is usually smaller)
            std::vector<float> B_local(k * n);
            if (mpi_rank_ == 0)
            {
                std::copy(B, B + k * n, B_local.begin());
            }
            MPI_Bcast(B_local.data(), k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

            // Ensure all ranks are synchronized before computation starts
            MPI_Barrier(MPI_COMM_WORLD);

            // Compute local portion
            const float *A_local = A + start_row * k;
            float *C_local = C + start_row * n;

            // Use all available cores per MPI rank for optimal performance
            int old_threads = openblas_get_num_threads();
            // Since we're running 1 MPI rank per socket, use all cores on this socket
            int mpi_threads = old_threads; // Use full thread count (28 cores per socket)
            openblas_set_num_threads(mpi_threads);

            if (mpi_rank_ == 0 && local_rows > 0)
            {
                std::cout << "MPI distribute: " << local_rows << " rows on rank 0, using "
                          << mpi_threads << " OpenBLAS threads" << std::endl;
            }

            CBLAS_TRANSPOSE trans_A = transpose_A ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE trans_B = transpose_B ? CblasTrans : CblasNoTrans;

            int lda = transpose_A ? local_rows : k;
            int ldb = transpose_B ? k : n;
            int ldc = n;

            cblas_sgemm(CblasRowMajor, trans_A, trans_B,
                        local_rows, n, k,
                        alpha, A_local, lda,
                        B_local.data(), ldb,
                        beta, C_local, ldc);

            openblas_set_num_threads(old_threads);

            // Ensure all ranks complete local computation before gathering
            MPI_Barrier(MPI_COMM_WORLD);

            // Gather results (allgather since everyone needs the full result)
            std::vector<int> recvcounts(mpi_size_);
            std::vector<int> displs(mpi_size_);

            for (int i = 0; i < mpi_size_; ++i)
            {
                int proc_start = i * rows_per_proc;
                int proc_end = std::min(proc_start + rows_per_proc, m);
                recvcounts[i] = std::max(0, proc_end - proc_start) * n;
                displs[i] = proc_start * n;
            }

            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                           C, recvcounts.data(), displs.data(), MPI_FLOAT,
                           MPI_COMM_WORLD);

            return true;
        }
    };

} // namespace llaminar