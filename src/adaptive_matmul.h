// Adaptive Matrix Multiplication Backend for Llaminar
//
// This implementation provides automatic selection between OpenBLAS and COSMA
// based on operation characteristics and empirical performance thresholds.
//
// Key features:
// - OpenBLAS for small operations and single token inference
// - COSMA for large operations and prefill scenarios (>4k tokens)
// - Automatic backend selection based on matrix dimensions
// - Performance monitoring and logging

#pragma once

#include <memory>
#include <map>
#include <limits>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cblas.h>
#include "tensors/tensor_factory.h"
#include "logger.h"
#include <mpi.h>
#include <cosma/cinterface.hpp>
#include <cosma/multiply.hpp>
#include <cosma/matrix.hpp>
#include <cosma/strategy.hpp>
#include <cosma/context.hpp>
#include "cosma_prefill_manager.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace llaminar
{

    enum class MatMulBackend
    {
        OPENBLAS, // For small operations and single token inference
        COSMA     // For large operations and prefill
    };

    class AdaptiveMatMulManager
    {
    private:
        // COSMA uses functional interface, no kernel object needed
        bool mpi_initialized_;
        int mpi_rank_, mpi_size_;

        // Simplified policy thresholds (empirically derived):
        // COSMA only provides benefit for large PREFILL (context build) operations.
        // Use COSMA IFF (is_prefill && seq_len >= PREFILL_COSMA_SEQ_THRESHOLD).
        // Otherwise always fall back to OpenBLAS which dominates for small/AR steps.
        static constexpr size_t PREFILL_COSMA_SEQ_THRESHOLD = 4096;  // Legacy crossover (now superseded by CosmaPrefillManager env gating)
        static constexpr size_t VOCAB_PROJECTION_THRESHOLD = 100000; // Always avoid COSMA on giant vocab projections

        // Performance monitoring
        struct OperationStats
        {
            size_t count = 0;
            double total_time_ms = 0.0;
            double min_time_ms = std::numeric_limits<double>::max();
            double max_time_ms = 0.0;

            void record(double time_ms)
            {
                count++;
                total_time_ms += time_ms;
                min_time_ms = std::min(min_time_ms, time_ms);
                max_time_ms = std::max(max_time_ms, time_ms);
            }

            double average() const { return count > 0 ? total_time_ms / count : 0.0; }
        };

        mutable std::map<std::string, OperationStats> backend_stats_;
        // Track last backend used (for test introspection)
        mutable MatMulBackend last_backend_ = MatMulBackend::OPENBLAS;

    public:
        AdaptiveMatMulManager()
        {
            int flag;
            MPI_Initialized(&flag);
            mpi_initialized_ = (flag != 0);

            if (mpi_initialized_)
            {
                MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
                MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);

                // COSMA uses functional interface - no initialization needed
            }
            else
            {
                mpi_rank_ = 0;
                mpi_size_ = 1;
            }
        }

        // Determine optimal backend based on operation characteristics
        MatMulBackend selectBackend(int m, int n, int k, bool is_prefill = false) const
        {
            // 1. If MPI not initialized or only one rank -> OpenBLAS.
            if (!mpi_initialized_ || mpi_size_ == 1)
            {
                return MatMulBackend::OPENBLAS;
            }

            auto &prefill_mgr = CosmaPrefillManager::instance();
            // Global force knob: override all normal policy except explicit disable env.
            if (prefill_mgr.force_cosma() && !std::getenv("ADAPTIVE_DISABLE_COSMA"))
            {
                return MatMulBackend::COSMA;
            }

            // 2. Skip COSMA for massive vocab projections (n very large)
            if (n > VOCAB_PROJECTION_THRESHOLD)
            {
                return MatMulBackend::OPENBLAS;
            }

            // 3. Use CosmaPrefillManager gating (env + world size) for prefill path
            if (is_prefill)
            {
                // Enforce explicit minimum sequence length for COSMA (policy): >= 4096 tokens
                if (m < static_cast<int>(PREFILL_COSMA_SEQ_THRESHOLD))
                {
                    return MatMulBackend::OPENBLAS; // force OpenBLAS below threshold
                }
                if (prefill_mgr.enabled_for(m))
                {
                    return MatMulBackend::COSMA;
                }
            }

            // 4. Everything else -> OpenBLAS.
            return MatMulBackend::OPENBLAS;
        }

        // High-level matrix multiplication interface
        // distributed_partition: set true when caller already partitions columns/rows
        // across MPI ranks (e.g. MPILinearKernel splits output features). In that case
        // we must NOT invoke COSMA because COSMA would expect the full global matrices
        // and perform a collective matmul producing inconsistent partial results.
        bool multiply(const float *A, const float *B, float *C,
                      int m, int n, int k,
                      bool transpose_A = false, bool transpose_B = false,
                      float alpha = 1.0f, float beta = 0.0f,
                      bool is_prefill = false,
                      bool distributed_partition = false)
        {
            MatMulBackend backend = MatMulBackend::OPENBLAS;
            if (!distributed_partition)
            {
                backend = selectBackend(m, n, k, is_prefill);
            }
            else
            {
                // Explicitly avoid COSMA for partitioned calls
                if (mpi_rank_ == 0)
                {
                    LOG_TRACE("AdaptiveMatMul: forcing OpenBLAS due to distributed partition (n=" << n << ")");
                }
            }
            last_backend_ = backend; // record decision

            if (mpi_rank_ == 0)
            {
                LOG_DEBUG("AdaptiveMatMul decision -> "
                          << (backend == MatMulBackend::COSMA ? "COSMA" : "OpenBLAS")
                          << " m=" << m << " n=" << n << " k=" << k
                          << (is_prefill ? " prefill" : " autoregressive"));
            }

            auto start = std::chrono::high_resolution_clock::now();
            bool success = false;

            switch (backend)
            {
            case MatMulBackend::COSMA:
            {
                // Use CosmaPrefillManager for large prefill path (Phase 1 correctness). Fallback to OpenBLAS on error.
                try
                {
                    auto &prefill_mgr = CosmaPrefillManager::instance();
                    if (!prefill_mgr.enabled_for(m))
                    {
                        // Gate not met (env threshold or world_size) -> fallback
                        success = multiply_openblas(A, B, C, m, n, k, transpose_A, transpose_B, alpha, beta);
                        break;
                    }
                    if (mpi_rank_ == 0)
                    {
                        LOG_DEBUG("AdaptiveMatMul: COSMA prefill manager engaged m=" << m << " n=" << n << " k=" << k);
                    }
                    if (transpose_A || transpose_B)
                    {
                        LOG_WARN("AdaptiveMatMul COSMA path: transpose not supported yet; falling back");
                        LOG_DEBUG("AdaptiveMatMul transpose fallback -> OpenBLAS start rank=" << mpi_rank_
                                                    << " m=" << m << " n=" << n << " k=" << k);
                        success = multiply_openblas(A, B, C, m, n, k, transpose_A, transpose_B, alpha, beta);
                        LOG_DEBUG("AdaptiveMatMul transpose fallback -> OpenBLAS complete rank=" << mpi_rank_
                                                    << " ok=" << success);
                        break;
                    }
                    // IMPORTANT: All COSMA matrices for a single GEMM must share the SAME strategy (m,n,k).
                    // Previous implementation used disparate strategies leading to wrong distributed mapping.
                    // Use single C strategy for all operands to match cosma::multiply expectations.
                    const auto &c_strat = prefill_mgr.unified_strategy(m, n, k);
                    WeightDescriptor wdesc{"adaptive", k, n, (int64_t)n, 1, 0, B};
                    auto A_view = prefill_mgr.convert_activation_operand(A, m, k, c_strat);
                    auto W_handle = prefill_mgr.load_weight_operand(wdesc, c_strat);
                    auto C_view = prefill_mgr.matmul(A_view, W_handle, m, k, n, false, alpha, beta);
                    prefill_mgr.to_row_major(C_view, C);
                    prefill_mgr.debug_compare_original(A_view, m, k, A);
                    prefill_mgr.debug_compare_original(W_handle.view, k, n, (const float *)B);
                    success = true;
                }
                catch (const std::exception &e)
                {
                    if (mpi_rank_ == 0)
                    {
                        LOG_ERROR("AdaptiveMatMul COSMA prefill manager path failed: " << e.what() << ", falling back to OpenBLAS");
                    }
                    success = multiply_openblas(A, B, C, m, n, k, transpose_A, transpose_B, alpha, beta);
                }
                break;
            }
            case MatMulBackend::OPENBLAS:
            {
                success = multiply_openblas(A, B, C, m, n, k, transpose_A, transpose_B, alpha, beta);
                break;
            }
            }

            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

            // Record performance statistics
            std::string backend_name = (backend == MatMulBackend::COSMA) ? "COSMA" : "OpenBLAS";
            backend_stats_[backend_name].record(time_ms);

            return success;
        }

        // Accessor for tests
        MatMulBackend last_backend() const { return last_backend_; }

        // Batched multiply: multiple A_i * shared B -> C_i. Always uses OpenBLAS (loop) for now.
        // Each A_i: (m_i x k), shared B: (k x n), C_i: (m_i x n). Intended for grouped projections (e.g., Q/K/V when laid out separately).
        bool multiply_batch(const float **A_list, const float *B, float **C_list,
                            const int *m_list, int batch_count, int n, int k,
                            bool is_prefill = false, float alpha = 1.0f, float beta = 0.0f)
        {
            if (batch_count <= 0)
                return true;
            // Policy: keep batches on OpenBLAS unless there is exactly one large op that meets COSMA threshold.
            if (batch_count == 1)
            {
                return multiply(A_list[0], B, C_list[0], m_list[0], n, k,
                                false, false, alpha, beta, is_prefill);
            }
            // Multi-op batch: sequential OpenBLAS calls (future: cblas_sgemm_batch optimization).
            for (int i = 0; i < batch_count; ++i)
            {
                if (!multiply_openblas(A_list[i], B, C_list[i], m_list[i], n, k,
                                       false, false, alpha, beta))
                {
                    return false;
                }
            }
            last_backend_ = MatMulBackend::OPENBLAS;
            return true;
        }

        // Get performance summary
        void printPerformanceSummary() const
        {
            if (mpi_rank_ != 0)
                return;

            LOG_INFO("\n=== Adaptive Backend Performance Summary ===");
            for (const auto &[backend, stats] : backend_stats_)
            {
                LOG_INFO(backend << ":");
                LOG_INFO("  Operations: " << stats.count);
                LOG_INFO("  Avg time: " << stats.average() << " ms");
                LOG_INFO("  Min time: " << stats.min_time_ms << " ms");
                LOG_INFO("  Max time: " << stats.max_time_ms << " ms");
                LOG_INFO("  Total time: " << stats.total_time_ms << " ms");
            }
        }

    private:
        bool multiply_cosma(const float *A, const float *B, float *C,
                            int m, int n, int k,
                            bool transpose_A, bool transpose_B,
                            float alpha, float beta)
        {
            // New COSMA integration using C interface multiply_using_layout.
            // Strategy: simple 1D row partition of A (m x k) and C (m x n) across ranks;
            // 1D partition of B along k dimension. This lets COSMA perform internal
            // reduction over split-k. After multiplication we Allgather C so every
            // rank has full result (to satisfy existing tests expecting replicated C).
            // NOTE: This is a correctness-first implementation; performance/true 2D
            // distribution & avoiding full replication are future optimizations.
            try
            {
                if (transpose_A || transpose_B)
                {
                    // Transpose support can be added by adjusting layouts + flags.
                    if (mpi_rank_ == 0)
                        LOG_WARN("COSMA path transpose not yet supported; falling back to OpenBLAS");
                    return multiply_openblas(A, B, C, m, n, k, transpose_A, transpose_B, alpha, beta);
                }

                // Helper lambdas to build row-split vectors (size blocks+1)
                auto build_split = [](int total, int parts)
                {
                    std::vector<int> split(parts + 1, total); // init with total
                    int base = total / parts;
                    int rem = total % parts;
                    int offset = 0;
                    for (int p = 0; p < parts; ++p)
                    {
                        split[p] = offset;
                        offset += base + (p < rem ? 1 : 0);
                    }
                    split[parts] = total;
                    return split;
                };

                // Row partition for A & C over m
                std::vector<int> a_rowsplit = build_split(m, mpi_size_);
                // Column split for A: single block across k
                std::vector<int> a_colsplit = {0, k};

                // B partitioned along its row dimension (k) across ranks to allow
                // COSMA to reduce partial products across split-k. Each rank holds
                // a contiguous slice of k rows of B.
                std::vector<int> b_rowsplit = build_split(k, mpi_size_);
                std::vector<int> b_colsplit = {0, n};

                // Row partition for C mirrors A (m rows)
                std::vector<int> c_rowsplit = a_rowsplit;
                std::vector<int> c_colsplit = {0, n};

                // Owners arrays (rowblocks * colblocks). Each block owned by its rank if non-empty.
                // Owners arrays encode which rank owns each global block.
                // A and C: one block row per rank (row-split), single column block.
                std::vector<int> a_owners(mpi_size_ * 1, 0);
                std::vector<int> c_owners(mpi_size_ * 1, 0);
                for (int r = 0; r < mpi_size_; ++r)
                {
                    a_owners[r] = r; // block (r,0) owned by rank r
                    c_owners[r] = r; // block (r,0) owned by rank r
                }
                // B owners: one block row per rank (k-split)
                std::vector<int> b_owners(mpi_size_ * 1, 0);
                for (int r = 0; r < mpi_size_; ++r)
                    b_owners[r] = r;

                // NOTE: COSMA assumes column-major storage (Fortran-style). Our tensors are
                // row-major. For correctness we create temporary column-major buffers for
                // the participating local slices (A_r, B, C_r), perform multiplication, then
                // convert back to row-major for gathering. This adds overhead but isolates
                // correctness issues; future optimization can introduce native row-major
                // support via transposition flags or custom layouts.

                // Local slice extents for this rank
                int a_row_start = a_rowsplit[mpi_rank_];
                int a_row_end = a_rowsplit[mpi_rank_ + 1];
                int local_rows_A = a_row_end - a_row_start; // also local_rows_C
                int c_row_start = a_row_start;              // identical partition
                int c_row_end = a_row_end;

                // Allocate / build column-major temporaries
                std::vector<float> local_A_colmaj;
                std::vector<float> local_C_colmaj;
                std::vector<float> local_B_colmaj; // only rank 0

                // Build local block descriptors
                std::vector<block> a_local_blocks;
                std::vector<block> b_local_blocks;
                std::vector<block> c_local_blocks;

                if (local_rows_A > 0)
                {
                    local_A_colmaj.resize(static_cast<size_t>(local_rows_A) * k);
                    const float *A_slice = A + static_cast<size_t>(a_row_start) * k; // row-major source
                    // Row-major (i*k + j) -> Col-major (i + j*ld) with ld = local_rows_A
                    for (int j = 0; j < k; ++j)
                    {
                        for (int i = 0; i < local_rows_A; ++i)
                        {
                            local_A_colmaj[static_cast<size_t>(i) + static_cast<size_t>(j) * local_rows_A] = A_slice[static_cast<size_t>(i) * k + j];
                        }
                    }
                    block blkA{(void *)local_A_colmaj.data(), local_rows_A, mpi_rank_, 0};
                    a_local_blocks.push_back(blkA);
                }

                // Local B slice for this rank
                int b_row_start = b_rowsplit[mpi_rank_];
                int b_row_end = b_rowsplit[mpi_rank_ + 1];
                int local_rows_B = b_row_end - b_row_start; // local k-slice
                if (local_rows_B > 0)
                {
                    local_B_colmaj.resize(static_cast<size_t>(local_rows_B) * n);
                    const float *B_slice = B + static_cast<size_t>(b_row_start) * n;
                    for (int j = 0; j < n; ++j)
                    {
                        for (int i = 0; i < local_rows_B; ++i)
                        {
                            local_B_colmaj[static_cast<size_t>(i) + static_cast<size_t>(j) * local_rows_B] = B_slice[static_cast<size_t>(i) * n + j];
                        }
                    }
                    block blkB{(void *)local_B_colmaj.data(), local_rows_B, mpi_rank_, 0};
                    b_local_blocks.push_back(blkB);
                }

                if (local_rows_A > 0)
                {
                    local_C_colmaj.resize(static_cast<size_t>(local_rows_A) * n);
                    float *C_slice = C + static_cast<size_t>(c_row_start) * n; // row-major destination slice
                    if (beta != 0.0f)
                    {
                        // Populate existing C into column-major buffer for beta scaling
                        for (int j = 0; j < n; ++j)
                        {
                            for (int i = 0; i < local_rows_A; ++i)
                            {
                                local_C_colmaj[static_cast<size_t>(i) + static_cast<size_t>(j) * local_rows_A] = C_slice[static_cast<size_t>(i) * n + j];
                            }
                        }
                    }
                    else
                    {
                        std::fill(local_C_colmaj.begin(), local_C_colmaj.end(), 0.0f);
                    }
                    block blkC{(void *)local_C_colmaj.data(), local_rows_A, mpi_rank_, 0};
                    c_local_blocks.push_back(blkC);
                }

                // Build layout structs
                layout layout_a{mpi_size_, 1, a_rowsplit.data(), a_colsplit.data(), a_owners.data(), (int)a_local_blocks.size(), a_local_blocks.data()};
                layout layout_b{mpi_size_, 1, b_rowsplit.data(), b_colsplit.data(), b_owners.data(), (int)b_local_blocks.size(), b_local_blocks.data()};
                layout layout_c{mpi_size_, 1, c_rowsplit.data(), c_colsplit.data(), c_owners.data(), (int)c_local_blocks.size(), c_local_blocks.data()};

                const char transa = 'N';
                const char transb = 'N';
                if (mpi_rank_ == 0)
                {
                    LOG_DEBUG("COSMA multiply_using_layout m=" << m << " n=" << n << " k=" << k
                                                               << " local_rows=" << (a_rowsplit[mpi_rank_ + 1] - a_rowsplit[mpi_rank_])
                                                               << " blocks(A)=" << layout_a.rowblocks << "x" << layout_a.colblocks
                                                               << " blocks(B)=" << layout_b.rowblocks << "x" << layout_b.colblocks
                                                               << " blocks(C)=" << layout_c.rowblocks << "x" << layout_c.colblocks
                                                               << " alpha=" << alpha << " beta=" << beta);
                }
                MPI_Barrier(MPI_COMM_WORLD);
                smultiply_using_layout(MPI_COMM_WORLD, &transa, &transb, &alpha, &layout_a, &layout_b, &beta, &layout_c);
                MPI_Barrier(MPI_COMM_WORLD);

                // Convert local C slice back to row-major prior to gather
                if (local_rows_A > 0)
                {
                    float *C_slice = C + static_cast<size_t>(c_row_start) * n;
                    for (int j = 0; j < n; ++j)
                    {
                        for (int i = 0; i < local_rows_A; ++i)
                        {
                            C_slice[static_cast<size_t>(i) * n + j] = local_C_colmaj[static_cast<size_t>(i) + static_cast<size_t>(j) * local_rows_A];
                        }
                    }
                }

                // Gather all C row slices so each rank has full matrix (tests expect replicated C)
                std::vector<int> recvcounts(mpi_size_, 0);
                std::vector<int> displs(mpi_size_, 0);
                for (int r = 0; r < mpi_size_; ++r)
                {
                    int rs = a_rowsplit[r];
                    int re = a_rowsplit[r + 1];
                    recvcounts[r] = (re - rs) * n;
                }
                int offset = 0;
                for (int r = 0; r < mpi_size_; ++r)
                {
                    displs[r] = offset;
                    offset += recvcounts[r];
                }
                int c_slice_elems = (c_row_end - c_row_start) * n;
                float *c_slice_ptr = (local_rows_A > 0) ? (C + static_cast<size_t>(c_row_start) * n) : nullptr;
                MPI_Allgatherv(c_slice_ptr, c_slice_elems, MPI_FLOAT,
                               C, recvcounts.data(), displs.data(), MPI_FLOAT,
                               MPI_COMM_WORLD);
                return true;
            }
            catch (const std::exception &e)
            {
                if (mpi_rank_ == 0)
                {
                    LOG_ERROR("COSMA layout multiply failed, falling back to OpenBLAS: " << e.what());
                }
                return multiply_openblas(A, B, C, m, n, k, transpose_A, transpose_B, alpha, beta);
            }
        }

    private:
        // Distributed OpenBLAS implementation for demonstration
        bool multiply_distributed_openblas(const float *A, const float *B, float *C,
                                           int m, int n, int k,
                                           bool transpose_A, bool transpose_B,
                                           float alpha, float beta)
        {

            // Simple distributed matrix multiplication using OpenBLAS on each rank
            // Each rank processes its portion of the data

            int rank = mpi_rank_;
            int size = mpi_size_;

            // For demonstration, distribute work by rows of A and C
            int rows_per_rank = (m + size - 1) / size;
            int start_row = rank * rows_per_rank;
            int end_row = std::min(start_row + rows_per_rank, m);
            int local_rows = std::max(0, end_row - start_row);

            if (local_rows > 0)
            {
                // Each rank computes its portion using OpenBLAS
                const float *A_local = A + start_row * k;
                float *C_local = C + start_row * n;

                // Use single-threaded OpenBLAS to avoid conflicts with MPI
                openblas_set_num_threads(1);

                CBLAS_TRANSPOSE trans_A = transpose_A ? CblasTrans : CblasNoTrans;
                CBLAS_TRANSPOSE trans_B = transpose_B ? CblasTrans : CblasNoTrans;

                int lda = transpose_A ? local_rows : k;
                int ldb = transpose_B ? k : n;
                int ldc = n;

                cblas_sgemm(CblasRowMajor, trans_A, trans_B,
                            local_rows, n, k,
                            alpha, A_local, lda,
                            B, ldb,
                            beta, C_local, ldc);
            }

            // Synchronize to ensure all ranks complete
            MPI_Barrier(MPI_COMM_WORLD);

            return true;
        }

        bool multiply_openblas(const float *A, const float *B, float *C,
                               int m, int n, int k,
                               bool transpose_A, bool transpose_B,
                               float alpha, float beta)
        {
            try
            {
                // Adaptive threading: allow multi-thread for large standalone ops unless user overrides.
                // Priority: explicit env > heuristic > 1 (fallback). Heuristic keeps small ops single-threaded.
                static int env_threads = []()
                {
                    // Priority: LLAMINAR_OPENBLAS_THREADS > OPENBLAS_NUM_THREADS
                    const char *v1 = std::getenv("LLAMINAR_OPENBLAS_THREADS");
                    const char *v2 = std::getenv("OPENBLAS_NUM_THREADS");
                    const char *v = v1 ? v1 : v2;
                    if (!v)
                        return -1; // no explicit override
                    int t = std::atoi(v);
                    return t < 1 ? 1 : t;
                }();
                int threads = 1;
                if (env_threads > 0)
                {
                    threads = env_threads;
                }
                else
                {
                    // Rough flop proxy; promote to multi-thread only for large matrices.
                    long double work = static_cast<long double>(m) * n * k; // ~multiplications
                    // Thresholds chosen empirically; adjust if needed.
                    if (work >= 25000000.0L)
                    { // 25M mults (~50 MFLOP) promote to multi-thread
#ifdef _OPENMP
                        int max_t = omp_get_max_threads();
                        int ranks = mpi_size_ > 0 ? mpi_size_ : 1;
                        // Simple per-rank cap: divide available threads among ranks (at least 1)
                        int per_rank = std::max(1, max_t / ranks);
                        // If problem extremely large (heuristic) and per_rank small, allow slight oversub (up to 2x) for single-rank runs only
                        threads = per_rank;
#else
                        threads = 4; // reasonable default if OpenMP not present
#endif
                    }
                }
                openblas_set_num_threads(threads);
                if (threads > 1 && mpi_rank_ == 0)
                {
                    LOG_TRACE("AdaptiveMatMul OpenBLAS using " << threads << " threads (m=" << m << ", n=" << n << ", k=" << k << ")");
                }

                CBLAS_TRANSPOSE trans_A = transpose_A ? CblasTrans : CblasNoTrans;
                CBLAS_TRANSPOSE trans_B = transpose_B ? CblasTrans : CblasNoTrans;

                int lda = transpose_A ? m : k; // RowMajor rule: lda = (transA? M : K)
                int ldb = transpose_B ? k : n; // RowMajor rule: ldb = (transB? K : N)
                int ldc = n;

                cblas_sgemm(CblasRowMajor, trans_A, trans_B,
                            m, n, k,
                            alpha, A, lda,
                            B, ldb,
                            beta, C, ldc);
                return true;
            }
            catch (const std::exception &e)
            {
                if (mpi_rank_ == 0)
                {
                    LOG_ERROR("OpenBLAS matmul failed: " << e.what());
                }
                return false;
            }
        }
    };

    // Global convenience functions for easy integration
    inline bool adaptiveMatMul(const float *A, const float *B, float *C,
                               int m, int n, int k,
                               bool is_prefill = false,
                               bool distributed_partition = false,
                               bool transpose_A = false,
                               bool transpose_B = false,
                               float alpha = 1.0f,
                               float beta = 0.0f)
    {
        static AdaptiveMatMulManager manager;
        return manager.multiply(A, B, C, m, n, k,
                                transpose_A, transpose_B,
                                alpha, beta,
                                is_prefill,
                                distributed_partition);
    }

    inline bool adaptive_matmul(const float *A, const float *B, float *C,
                                int m, int n, int k, bool is_prefill = false,
                                bool distributed_partition = false)
    {
        return adaptiveMatMul(A, B, C, m, n, k, is_prefill, distributed_partition);
    }

} // namespace llaminar