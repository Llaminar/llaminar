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

// Include MKL backend if available (BEFORE OpenBLAS to avoid header conflicts)
#ifdef HAVE_MKL
#include "backends/MKLBackend.h"
#endif

// Include OpenBLAS cblas.h AFTER MKL headers (if both present)
// OpenBLAS common.h defines FLOAT/DOUBLE macros that conflict with MPI/OpenMP
#include <cblas.h>

// Undefine OpenBLAS internal macros that conflict with MPI
#ifdef FLOAT
#undef FLOAT
#endif
#ifdef DOUBLE
#undef DOUBLE
#endif
#ifdef COMPLEX
#undef COMPLEX
#endif
#ifdef COMPLEX16
#undef COMPLEX16
#endif

#include <memory>
#include <map>
#include <limits>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <cctype>
#include "tensors/TensorFactory.h"
#include "QuantizedGemm.h"
#include "Logger.h"
#include "utils/BFloat16.h"
#include "utils/CpuFeatures.h"
#include <mpi.h>
#include <cosma/cinterface.hpp>
#include <cosma/multiply.hpp>
#include <cosma/matrix.hpp>
#include <cosma/strategy.hpp>
#include <cosma/context.hpp>
#include "CosmaPrefillManager.h"
#include "utils/DebugEnv.h"     // Added for centralized debug environment access (debugEnv())
#include "utils/PerfCounters.h" // performance instrumentation
#ifdef _OPENMP
#include <omp.h>
#endif

#include "MatmulBackendSelection.h" // For MatMulBackend enum and MatMulBackendSelector

namespace llaminar
{
    // Note: MatMulBackend enum now defined in MatmulBackendSelection.h
    // Legacy MULTI_THREADED_OPENBLAS/COSMA values used directly

    class AdaptiveMatMulManager
    {
    private:
        // COSMA uses functional interface, no kernel object needed
        bool mpi_initialized_;
        int mpi_rank_, mpi_size_;

        static bool env_flag_enabled(const char *value)
        {
            if (!value)
            {
                return false;
            }
            std::string token;
            token.reserve(std::strlen(value));
            for (const char *p = value; *p; ++p)
            {
                unsigned char ch = static_cast<unsigned char>(*p);
                if (!std::isspace(ch))
                {
                    token.push_back(static_cast<char>(std::tolower(ch)));
                }
            }
            if (token.empty())
            {
                return true;
            }
            return !(token == "0" || token == "false" || token == "off" || token == "no");
        }

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
        mutable MatMulBackend last_backend_ = MatMulBackend::MULTI_THREADED_OPENBLAS;

        // GEMM strategy cache: maps TensorBase* to cached ITensorGemm instance
        // Avoids repeated heap allocation and virtual calls for same tensor
        mutable std::map<const TensorBase *, std::unique_ptr<ITensorGemm>> gemm_cache_;

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
                return MatMulBackend::MULTI_THREADED_OPENBLAS;
            }

            // Access centralized debug environment snapshot
            const bool adaptive_disabled = debugEnv().adaptive.disable_cosma;
            const bool cosma_disabled = debugEnv().cosma.disable;

            if (adaptive_disabled || cosma_disabled)
            {
                return MatMulBackend::MULTI_THREADED_OPENBLAS;
            }

            auto &prefill_mgr = CosmaPrefillManager::instance();
            // Global force knob: override all normal policy except explicit disable env.
            if (prefill_mgr.force_cosma())
            {
                return MatMulBackend::COSMA;
            }

            // 2. Skip COSMA for massive vocab projections (n very large)
            if (n > VOCAB_PROJECTION_THRESHOLD)
            {
                return MatMulBackend::MULTI_THREADED_OPENBLAS;
            }

            // 3. Use CosmaPrefillManager gating (env + world size) for prefill path
            if (is_prefill)
            {
                // Enforce explicit minimum sequence length for COSMA (policy): >= 4096 tokens
                if (m < static_cast<int>(PREFILL_COSMA_SEQ_THRESHOLD))
                {
                    return MatMulBackend::MULTI_THREADED_OPENBLAS; // force OpenBLAS below threshold
                }
                if (prefill_mgr.enabled_for(m))
                {
                    return MatMulBackend::COSMA;
                }
            }

            // 4. Everything else -> OpenBLAS.
            return MatMulBackend::MULTI_THREADED_OPENBLAS;
        }

        // High-level matrix multiplication interface
        // distributed_partition: set true when caller already partitions columns/rows
        // across MPI ranks (e.g. MPILinearOperator splits output features). In that case
        // we must NOT invoke COSMA because COSMA would expect the full global matrices
        // and perform a collective matmul producing inconsistent partial results.
        bool multiply(const float *A, const float *B, float *C,
                      int m, int n, int k,
                      bool transpose_A = false, bool transpose_B = false,
                      float alpha = 1.0f, float beta = 0.0f,
                      bool is_prefill = false,
                      bool distributed_partition = false)
        {
            MatMulBackend backend = MatMulBackend::MULTI_THREADED_OPENBLAS;
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
            case MatMulBackend::MULTI_THREADED_OPENBLAS:
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

        // Accessor for GEMM cache (allows free function to cache strategies)
        std::map<const TensorBase *, std::unique_ptr<ITensorGemm>> &getGemmCache() const
        {
            return gemm_cache_;
        }

        /**
         * @brief BF16×BF16→FP32 matrix multiplication using OpenBLAS cblas_sbgemm
         *
         * @param A Input matrix A (m×k) in FP32 - will be converted to BF16 internally
         * @param B_bf16 Weight matrix B (k×n) in BF16 format
         * @param C Output matrix C (m×n) in FP32
         * @param m Number of rows in A and C
         * @param n Number of columns in B and C
         * @param k Number of columns in A and rows in B
         * @param transpose_B Whether B is transposed (currently only transpose_B=true supported for weights)
         * @param alpha Scalar alpha (default 1.0)
         * @param beta Scalar beta (default 0.0)
         * @return true on success, false on failure
         *
         * @note OpenBLAS sbgemm performs BF16×BF16 multiply with FP32 accumulation.
         *       Input A is converted from FP32→BF16 before computation.
         *       This exploits 2× bandwidth reduction for weights while preserving accuracy.
         */
        bool multiplyBF16(const float *A, const bfloat16 *B_bf16, float *C,
                          int m, int n, int k,
                          bool transpose_B = true,
                          float alpha = 1.0f, float beta = 0.0f)
        {
            // Check if BF16 GEMM is enabled
            if (!debugEnv().quant.bf16_gemm)
            {
                if (mpi_rank_ == 0)
                {
                    LOG_DEBUG("BF16 GEMM disabled (LLAMINAR_QUANT_BF16_GEMM=0), falling back to FP32 expansion");
                }
                return false; // Caller should fallback to FP32 expansion
            }

#ifdef HAVE_MKL
            // Intel MKL is the default BF16 backend when compiled in (no CPU feature requirements)
            // Only skip MKL if explicitly disabled via LLAMINAR_QUANT_BF16_PREFER_MKL=0
            bool use_mkl = true;
            if (const char *prefer_env = std::getenv("LLAMINAR_QUANT_BF16_PREFER_MKL"))
            {
                std::string val(prefer_env);
                if (val == "0" || val == "false" || val == "off")
                {
                    use_mkl = false;
                    if (mpi_rank_ == 0)
                    {
                        LOG_DEBUG("MKL explicitly disabled via LLAMINAR_QUANT_BF16_PREFER_MKL=0");
                    }
                }
            }

            if (use_mkl)
            {
                if (mpi_rank_ == 0)
                {
                    LOG_DEBUG("Using MKL BF16 GEMM (default): m=" << m << " n=" << n << " k=" << k);
                }

                bool mkl_ok = mkl_multiply_bf16(
                    A, B_bf16, C, m, n, k,
                    alpha, beta,
                    false,       // transpose_A (always false for activations)
                    transpose_B, // transpose_B (weights)
                    false        // validate_inputs (debug only)
                );

                if (mkl_ok)
                {
                    if (mpi_rank_ == 0)
                    {
                        LOG_DEBUG("MKL BF16 GEMM succeeded");
                    }
                    return true;
                }

                // MKL failed, log warning and try OpenBLAS fallback
                if (mpi_rank_ == 0)
                {
                    LOG_WARN("MKL BF16 GEMM failed, falling back to OpenBLAS");
                }
            }
#endif

            // OpenBLAS fallback path
            // NOTE: Original defensive check removed after verifying OpenBLAS v0.3.26
            //       BF16 emulation works correctly on Cascade Lake (Oct 20, 2025)
            // Previous code checked: if (!can_use_native_bf16_gemm()) { return false; }
            // Test results: cblas_sbgemm works without NaN on all matrix sizes
            // See: changelog/2025-10-20-openblas-bf16-bug-investigation.md

            // Optionally warn if using software emulation (informational only)
            if (mpi_rank_ == 0 && !can_use_native_bf16_gemm())
            {
                static bool logged_once = false;
                if (!logged_once)
                {
                    LOG_INFO("CPU lacks AVX512_BF16 - using OpenBLAS software BF16 emulation (verified working in v0.3.26)");
                    logged_once = true;
                }
            }

            if (mpi_rank_ == 0)
            {
                LOG_DEBUG("AdaptiveMatMul::multiplyBF16 m=" << m << " n=" << n << " k=" << k
                                                            << " transpose_B=" << transpose_B);
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            try
            {
                // Convert input A from FP32 to BF16
                // Use raw uint16_t buffer for OpenBLAS compatibility (avoids struct casting issues)
                std::vector<uint16_t> A_bf16_raw((size_t)m * (size_t)k);

#ifdef _OPENMP
                size_t total = (size_t)m * (size_t)k;
#pragma omp parallel for if (total > 32768) schedule(static)
                for (size_t idx = 0; idx < total; ++idx)
                {
                    A_bf16_raw[idx] = bfloat16::from_float(A[idx]).data;
                }
#else
                for (size_t idx = 0; idx < (size_t)m * (size_t)k; ++idx)
                {
                    A_bf16_raw[idx] = bfloat16::from_float(A[idx]).data;
                }
#endif

                // Also convert B_bf16 to raw uint16_t buffer
                size_t B_size = transpose_B ? ((size_t)n * (size_t)k) : ((size_t)k * (size_t)n);
                std::vector<uint16_t> B_bf16_raw(B_size);
#ifdef _OPENMP
#pragma omp parallel for if (B_size > 32768) schedule(static)
                for (size_t idx = 0; idx < B_size; ++idx)
                {
                    B_bf16_raw[idx] = B_bf16[idx].data;
                }
#else
                for (size_t idx = 0; idx < B_size; ++idx)
                {
                    B_bf16_raw[idx] = B_bf16[idx].data;
                }
#endif

                // Adaptive threading (same logic as multiply_openblas)
                static int env_threads = []()
                {
                    int t = llaminar::debugEnv().cosma.forced_openblas_threads;
                    return t > 0 ? t : -1;
                }();
                int threads = 1;
                if (env_threads > 0)
                {
                    threads = env_threads;
                }
                else
                {
                    long double work = static_cast<long double>(m) * n * k;
                    if (work >= 25000000.0L)
                    {
#ifdef _OPENMP
                        int max_t = omp_get_max_threads();
                        int ranks = mpi_size_ > 0 ? mpi_size_ : 1;
                        threads = std::max(1, max_t / ranks);
#else
                        threads = 4;
#endif
                    }
                }
                openblas_set_num_threads(threads);

                // Call cblas_sbgemm
                // Note: OpenBLAS bfloat16 is uint16_t, our bfloat16 is a struct with uint16_t data member
                // Cast via reinterpret_cast to match OpenBLAS signature

                CBLAS_TRANSPOSE trans_B = transpose_B ? CblasTrans : CblasNoTrans;
                int lda = k;                   // A is row-major (m×k)
                int ldb = transpose_B ? n : k; // B is (k×n) with transpose
                int ldc = n;                   // C is row-major (m×n)

                if (mpi_rank_ == 0)
                {
                    LOG_DEBUG("cblas_sbgemm params: m=" << m << " n=" << n << " k=" << k
                                                        << " transpose_B=" << transpose_B << " trans_B=" << (trans_B == CblasTrans ? "Trans" : "NoTrans")
                                                        << " lda=" << lda << " ldb=" << ldb << " ldc=" << ldc
                                                        << " A_size=" << A_bf16_raw.size() << " B_size=" << B_bf16_raw.size());

                    // Validate inputs don't contain NaN/Inf in BF16 form
                    bool a_valid = true, b_valid = true;
                    for (size_t i = 0; i < std::min((size_t)10, A_bf16_raw.size()); ++i)
                    {
                        uint16_t val = A_bf16_raw[i];
                        if ((val & 0x7FFF) == 0x7F80)
                        { // NaN/Inf check for BF16
                            a_valid = false;
                            break;
                        }
                    }
                    for (size_t i = 0; i < std::min((size_t)10, B_bf16_raw.size()); ++i)
                    {
                        uint16_t val = B_bf16_raw[i];
                        if ((val & 0x7FFF) == 0x7F80)
                        { // NaN/Inf check for BF16
                            b_valid = false;
                            break;
                        }
                    }
                    LOG_DEBUG("Input validation: A_valid=" << a_valid << " B_valid=" << b_valid);
                }

                // OpenBLAS sbgemm: BF16×BF16→FP32
                // Use raw uint16_t buffers (no struct casting needed)
                cblas_sbgemm(CblasRowMajor, CblasNoTrans, trans_B,
                             m, n, k,
                             alpha, reinterpret_cast<const ::bfloat16 *>(A_bf16_raw.data()), lda,
                             reinterpret_cast<const ::bfloat16 *>(B_bf16_raw.data()), ldb,
                             beta, C, ldc);

                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

                if (mpi_rank_ == 0)
                {
                    LOG_DEBUG("BF16 GEMM completed in " << ms << "ms");
                }

                perfCounters().record_matmul(m, n, k, ms, (int)MatMulBackend::MULTI_THREADED_OPENBLAS, false);
                return true;
            }
            catch (const std::exception &e)
            {
                if (mpi_rank_ == 0)
                {
                    LOG_ERROR("BF16 GEMM failed: " << e.what() << ", falling back to FP32");
                }
                return false;
            }
        }

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
            last_backend_ = MatMulBackend::MULTI_THREADED_OPENBLAS;
            return true;
        }

        // Get performance summary
        void printPerformanceSummary() const
        {
            if (mpi_rank_ != 0)
                return;

            if (llaminar::debugEnv().performance.enable)
            {
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
        }

    private:
        bool multiply_cosma(const float *A, const float *B, float *C,
                            int m, int n, int k,
                            bool transpose_A, bool transpose_B,
                            float alpha, float beta)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
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
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
                perfCounters().record_matmul(m, n, k, ms, (int)MatMulBackend::COSMA, true);
                return true;
            }
            catch (const std::exception &e)
            {
                if (mpi_rank_ == 0)
                {
                    LOG_ERROR("COSMA layout multiply failed, falling back to OpenBLAS: " << e.what());
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
                perfCounters().record_matmul(m, n, k, ms, (int)MatMulBackend::COSMA, true);
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
            // Early return for empty tensors (e.g., rank with no tokens to process)
            if (m == 0 || n == 0 || k == 0)
            {
                LOG_TRACE("[OPENBLAS_SKIP] Empty tensor: m=" << m << " n=" << n << " k=" << k << ", skipping computation on rank " << mpi_rank_);
                return true; // Success - nothing to compute
            }

            // DEBUG: Entry logging
            LOG_TRACE("[OPENBLAS_ENTRY] rank=" << mpi_rank_ << " m=" << m << " n=" << n << " k=" << k
                                               << " transpose_A=" << transpose_A << " transpose_B=" << transpose_B);

            auto t0 = std::chrono::high_resolution_clock::now();
            try
            {
                // Adaptive threading: allow multi-thread for large standalone ops unless user overrides.
                // Priority: explicit env > heuristic > 1 (fallback). Heuristic keeps small ops single-threaded.
                static int env_threads = []()
                { int t = llaminar::debugEnv().cosma.forced_openblas_threads; return t>0? t : -1; }();
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

                // DEBUG: Log cblas parameters for Q/K/V projections (BOTH ranks for comparison)
                if (m == 4 && k == 896 && n == 448)
                {
                    LOG_DEBUG("[!!!ADAPTIVE_CBLAS!!!] rank " << mpi_rank_ << " Q/K/V projection:");
                    LOG_DEBUG("  m=" << m << " n=" << n << " k=" << k);
                    LOG_DEBUG("  transpose_A=" << transpose_A << " transpose_B=" << transpose_B);
                    LOG_DEBUG("  trans_A=" << (trans_A == CblasTrans ? "Trans" : "NoTrans")
                                           << " trans_B=" << (trans_B == CblasTrans ? "Trans" : "NoTrans"));
                    LOG_DEBUG("  lda=" << lda << " ldb=" << ldb << " ldc=" << ldc);
                    LOG_DEBUG("  A[0:5]: [" << A[0] << ", " << A[1] << ", " << A[2] << ", " << A[3] << ", " << A[4] << "]");
                    LOG_DEBUG("  B[0:5]: [" << B[0] << ", " << B[1] << ", " << B[2] << ", " << B[3] << ", " << B[4] << "]");
                }

                cblas_sgemm(CblasRowMajor, trans_A, trans_B,
                            m, n, k,
                            alpha, A, lda,
                            B, ldb,
                            beta, C, ldc);

                // DEBUG: Log output
                if (m == 4 && k == 896 && n == 448)
                {
                    LOG_DEBUG("  C[0:10]: [" << C[0] << ", " << C[1] << ", " << C[2] << ", " << C[3] << ", "
                                             << C[4] << ", " << C[5] << ", " << C[6] << ", " << C[7] << ", " << C[8] << ", " << C[9] << "]");
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
                perfCounters().record_matmul(m, n, k, ms, (int)MatMulBackend::MULTI_THREADED_OPENBLAS, false);
                return true;
            }
            catch (const std::exception &e)
            {
                if (mpi_rank_ == 0)
                {
                    LOG_ERROR("OpenBLAS matmul failed: " << e.what());
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
                perfCounters().record_matmul(m, n, k, ms, (int)MatMulBackend::MULTI_THREADED_OPENBLAS, false);
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

    /**
     * @brief Adaptive matrix multiplication with optional fused quantized GEMM
     *
     * This overload accepts a TensorBase* for the weight matrix and automatically
     * uses fused quantized GEMM if available, falling back to full decode + BLAS.
     *
     * @param A Input activation matrix [m, k] (FP32)
     * @param B_tensor Weight tensor (may be quantized)
     * @param C Output matrix [m, n] (FP32)
     * @param m Number of rows in A and C
     * @param n Number of columns in C (rows in B before transpose)
     * @param k Number of columns in A (columns in B)
     * @param is_prefill Whether this is a prefill operation
     * @param distributed_partition Whether weight is distributed across ranks
     * @param transpose_A Whether to transpose A
     * @param transpose_B Whether to transpose B (most common: true for [n,k] weights)
     * @param alpha Scaling factor for A @ B
     * @param beta Scaling factor for existing C
     *
     * @return true if operation succeeded, false otherwise
     */
    inline bool adaptiveMatMul(const float *A, const TensorBase *B_tensor, float *C,
                               int m, int n, int k,
                               bool is_prefill = false,
                               bool distributed_partition = false,
                               bool transpose_A = false,
                               bool transpose_B = false,
                               float alpha = 1.0f,
                               float beta = 0.0f)
    {
        if (!B_tensor)
        {
            LOG_ERROR("adaptiveMatMul: null B_tensor provided");
            return false;
        }

        // Try fused quantized GEMM path with caching
        static AdaptiveMatMulManager manager;
        auto &cache = manager.getGemmCache();

        // Check cache first
        auto it = cache.find(B_tensor);
        ITensorGemm *gemm = nullptr;

        if (it != cache.end())
        {
            gemm = it->second.get();
        }
        else
        {
            // Cache miss: create and store
            ITensorGemm *gemm_raw = B_tensor->createGemmRaw();
            if (gemm_raw)
            {
                cache[B_tensor] = std::unique_ptr<ITensorGemm>(gemm_raw);
                gemm = gemm_raw;
            }
        }

        if (gemm && gemm->supports(m, n, k))
        {
            LOG_DEBUG("Using tensor-specific GEMM: " << gemm->name());
            return gemm->multiply(A, C, m, n, k, transpose_B, alpha, beta);
        }

        // Fallback: full decode + BLAS
        const float *B = B_tensor->data();
        if (!B)
        {
            LOG_ERROR("adaptiveMatMul: B_tensor->data() returned null");
            return false;
        }

        return adaptiveMatMul(A, B, C, m, n, k,
                              is_prefill, distributed_partition,
                              transpose_A, transpose_B,
                              alpha, beta);
    }

    // FP16 weight convenience: expand FP16 weights to FP32 then call adaptiveMatMul.
    // Slab holds row-major (k x n) so no transpose needed.
    inline bool adaptiveMatMulFp16WeightsExpand(const float *A, const _Float16 *B_half, float *C,
                                                int m, int n, int k,
                                                float alpha = 1.0f, float beta = 0.0f,
                                                bool is_prefill = false,
                                                bool distributed_partition = false)
    {
        // Allocate FP32 buffer for weights; parallelize conversion for large slabs.
        std::vector<float> B_fp32((size_t)k * (size_t)n);
#ifdef _OPENMP
        size_t total = (size_t)k * (size_t)n;
#pragma omp parallel for if (total > 32768) schedule(static)
        for (size_t idx = 0; idx < total; ++idx)
        {
            B_fp32[idx] = (float)B_half[idx];
        }
#else
        for (size_t idx = 0; idx < (size_t)k * (size_t)n; ++idx)
            B_fp32[idx] = (float)B_half[idx];
#endif
        return adaptiveMatMul(A, B_fp32.data(), C, m, n, k, is_prefill, distributed_partition,
                              /*transpose_A*/ false, /*transpose_B*/ false, alpha, beta);
    }

    /**
     * @brief BF16 weight GEMM: BF16×BF16→FP32 using OpenBLAS cblas_sbgemm
     *
     * @param A Input activations (m×k) in FP32 - converted to BF16 internally
     * @param B_bf16 Weight matrix (k×n) in BF16 format (from slab cache)
     * @param C Output matrix (m×n) in FP32
     * @param m Number of rows in A and C
     * @param n Number of columns in B and C
     * @param k Number of columns in A and rows in B
     * @param alpha Scalar multiplier (default 1.0)
     * @param beta Scalar for C (default 0.0)
     * @param is_prefill Whether this is prefill (for logging)
     * @param distributed_partition Whether weights are already partitioned (for logging)
     * @param transpose_B Whether B needs transpose (default true for weight format)
     * @return true on success, false if BF16 disabled (caller should fallback)
     *
     * @note Requires LLAMINAR_QUANT_BF16_GEMM=1 environment variable
     * @note Only OpenBLAS backend supported currently (COSMA/MKL fallback to FP32)
     */
    inline bool adaptiveMatMulBF16(const float *A, const bfloat16 *B_bf16, float *C,
                                   int m, int n, int k,
                                   float alpha = 1.0f, float beta = 0.0f,
                                   bool is_prefill = false,
                                   bool distributed_partition = false,
                                   bool transpose_B = true)
    {
        static AdaptiveMatMulManager manager;
        return manager.multiplyBF16(A, B_bf16, C, m, n, k, transpose_B, alpha, beta);
    }

    inline bool adaptive_matmul(const float *A, const float *B, float *C,
                                int m, int n, int k, bool is_prefill = false,
                                bool distributed_partition = false)
    {
        return adaptiveMatMul(A, B, C, m, n, k, is_prefill, distributed_partition);
    }

} // namespace llaminar