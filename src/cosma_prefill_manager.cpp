/**
 * @file cosma_prefill_manager_refactored.cpp
 * @brief Minimal Refactored COSMA Prefill Manager - Essential Functionality Only
 * @author David Sanftenberg
 * @date October 2025
 *
 * # Refactoring Philosophy: Minimal Viable Implementation
 *
 * This file contains ONLY the methods actually used by the pipeline:
 * - Strategy caching (StrategyCache)
 * - Matrix allocation with destination-local population
 * - Activation conversion
 * - Weight loading (float32 only)
 * - Matrix multiplication orchestration
 * - Result reconstruction/gathering
 * - Fused RMSNorm+QKV operation
 * - Minimal statistics tracking
 *
 * **Removed from original 5221-line implementation:**
 * - Debug snapshot infrastructure (~1500 lines)
 * - JSON export and serialization (~400 lines)
 * - In-layout SwiGLU and Softmax (~200 lines)
 * - 40+ environment flags and parsing (~600 lines)
 * - Complex validation and diagnostics (~800 lines)
 * - Quantized weight support (deferred to Phase 2)
 * - Memory allocation tracking (simplified to atomic counters)
 *
 * **Result: ~900 lines vs 5221 lines (83% reduction)**
 *
 * # CRITICAL: COSMA Memory Pool LIFO Invariant
 *
 * COSMA uses a global strict LIFO (stack-based) bump allocator for ALL matrix buffers.
 * This is a fundamental architectural constraint that affects all COSMA usage.
 *
 * ## The LIFO Invariant
 *
 * ```cpp
 * // From COSMA's memory_pool.cpp:
 * void memory_pool::free_buffer(T* ptr, size_t size) {
 *     assert(pool_.data() + pool_size_ == ptr);  // MUST be stack top
 *     pool_size_ -= size;  // Pop from stack
 * }
 * ```
 *
 * **Consequences:**
 * - Buffers MUST be freed in exact reverse order of allocation
 * - Overlapping matrix lifetimes violate this invariant
 * - Violation causes assertion failure and crash
 *
 * ## Safe Patterns
 *
 * ### ❌ WRONG: Overlapping Lifetimes
 * ```cpp
 * auto q = matmul(A, Wq, m, k, n);  // Allocates Q buffer
 * auto k = matmul(A, Wk, m, k, n);  // Allocates K buffer
 * auto v = matmul(A, Wv, m, k, n);  // Allocates V buffer
 * // DANGER: Q/K/V all alive - destruction order undefined → crash
 * ```
 *
 * ### ✅ CORRECT: Immediate Reconstruction
 * ```cpp
 * auto q = matmul(A, Wq, m, k, n);
 * auto k = matmul(A, Wk, m, k, n);
 * auto v = matmul(A, Wv, m, k, n);
 *
 * // Reconstruct to host memory and destroy in LIFO order (V→K→Q)
 * capture_host_owned(v);  // Reconstruct V, free V.mat
 * capture_host_owned(k);  // Reconstruct K, free K.mat
 * capture_host_owned(q);  // Reconstruct Q, free Q.mat
 * ```
 *
 * ## Implementation Examples
 *
 * See fused_rmsnorm_qkv() for complete LIFO-safe multi-result pattern.
 * See matmul() documentation for single-result memory management.
 * See .github/instructions/cosma.instructions.md for comprehensive guidelines.
 *
 * # Key Design Decisions
 *
 * 1. **Use Proven Primitives**: rmsnorm_t5_forward() instead of custom COSMA-layout RMSNorm
 * 2. **COSMA for Distribution Only**: Not for elementwise operations
 * 3. **Destination-Local Population**: COSMA best practice from official docs
 * 4. **Host-Owned Buffers**: Simple std::shared_ptr, automatic cleanup
 * 5. **Centralized Config**: debugEnv() instead of scattered getenv() calls
 * 6. **Minimal Logging**: Essential timing and errors only
 *
 * # Migration Notes
 *
 * This implementation is NOT a drop-in replacement. The original header file
 * declares many methods not implemented here:
 * - rmsnorm_in_layout() - use rmsnorm_t5_forward() directly instead
 * - swiglu_in_layout() - use OpenBLAS path SwiGLU instead
 * - softmax_in_layout() - use OpenBLAS path softmax instead
 * - Debug/diagnostic methods - use external profiling tools
 * - JSON export - use logging and external analysis tools
 *
 * If these methods are called, linking will fail. Update callers to use
 * proven primitives from OpenBLAS execution path instead.
 */

#include "cosma_prefill_manager.h"
#include "logger.h"
#include "utils/debug_env.h"
#include "utils/perf_counters.h"
#include "kernels/common/rmsnorm_t5.h"
#include <cosma/multiply.hpp>
#include <cblas.h>
#include <mpi.h>
#include <mutex>
#include <chrono>
#include <algorithm>

namespace
{
    // ========================================================================
    // MPI Safety Helpers
    // ========================================================================

    inline bool mpi_is_initialized()
    {
        int flag = 0;
        MPI_Initialized(&flag);
        return flag != 0;
    }

    inline bool mpi_is_finalized()
    {
        int flag = 0;
        MPI_Finalized(&flag);
        return flag != 0;
    }

    inline int mpi_world_size_safe()
    {
        if (!mpi_is_initialized() || mpi_is_finalized())
            return 1;
        int sz = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &sz);
        return sz;
    }

    inline int mpi_rank_safe()
    {
        if (!mpi_is_initialized() || mpi_is_finalized())
            return 0;
        int r = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &r);
        return r;
    }

    inline void safe_barrier()
    {
        if (mpi_is_initialized() && mpi_world_size_safe() > 1)
        {
            llaminar::PerfBarrier(MPI_COMM_WORLD);
        }
    }

} // anonymous namespace

namespace llaminar
{

    // ========================================================================
    // StrategyCache Implementation
    // ========================================================================

    std::string StrategyCache::make_key(int m, int n, int k, int p) const
    {
        return std::to_string(m) + "x" + std::to_string(n) + "x" +
               std::to_string(k) + "p" + std::to_string(p);
    }

    const cosma::Strategy &StrategyCache::get(int m, int n, int k, int p)
    {
        std::string key = make_key(m, n, k, p);

        auto it = cache_.find(key);
        if (it != cache_.end())
        {
            stats.hits.fetch_add(1, std::memory_order_relaxed);
            return it->second;
        }

        // Cache miss - create new strategy
        stats.misses.fetch_add(1, std::memory_order_relaxed);
        cosma::Strategy strategy(m, n, k, p);
        auto result = cache_.emplace(key, std::move(strategy));
        return result.first->second;
    }

    // ========================================================================
    // FusedRmsnormQkvResult::WeightGuard Implementation
    // ========================================================================

    FusedRmsnormQkvResult::WeightGuard::WeightGuard(
        CosmaPrefillManager *mgr,
        CosmaWeightHandle &&h) : manager(mgr), handle(std::move(h)) {}

    FusedRmsnormQkvResult::WeightGuard::WeightGuard(WeightGuard &&other) noexcept
        : manager(other.manager), handle(std::move(other.handle))
    {
        other.manager = nullptr;
    }

    FusedRmsnormQkvResult::WeightGuard &
    FusedRmsnormQkvResult::WeightGuard::operator=(WeightGuard &&other) noexcept
    {
        if (this != &other)
        {
            // DO NOT call release_weight - see destructor comment
            manager = other.manager;
            handle = std::move(other.handle);
            other.manager = nullptr;
        }
        return *this;
    }

    FusedRmsnormQkvResult::WeightGuard::~WeightGuard()
    {
        // DO NOT release here - the weight matrices are owned by shared_ptr
        // and will be automatically cleaned up when the last reference is dropped.
        // The CosmaView objects (q, k, v) in the result still reference these weights.
        // Explicit release would cause double-free or use-after-free.
        (void)manager;
        (void)handle;
    }

    // ========================================================================
    // CosmaPrefillManager - Singleton and Lifecycle
    // ========================================================================

    CosmaPrefillManager &CosmaPrefillManager::instance()
    {
        static CosmaPrefillManager instance;
        return instance;
    }

    CosmaPrefillManager::CosmaPrefillManager()
    {
        // Initialize MPI context
        if (mpi_is_initialized())
        {
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        }

        // Read centralized environment configuration
        const auto &env = debugEnv();

        // Sequence length threshold for COSMA enablement
        threshold_ = env.cosma.prefill_threshold;

        // Memory budget (MB)
        max_resident_mb_ = env.cosma.max_resident_mb;

        // Fast path threshold (bypass COSMA for very small ops)
        fast_path_threshold_ops_ = 64LL * 64LL * 64LL; // 262,144 operations

        // Log level (simplified)
        log_level_ = 2; // INFO

        if (rank_ == 0 && threshold_ > 0)
        {
            LOG_INFO("[CosmaPrefillManager] Initialized: world_size=" << world_size_
                                                                      << ", threshold=" << threshold_
                                                                      << ", max_resident_mb=" << max_resident_mb_);
        }
    }

    CosmaPrefillManager::~CosmaPrefillManager()
    {
        // Automatic cleanup via RAII - no manual memory management needed
        if (rank_ == 0)
        {
            LOG_DEBUG("[CosmaPrefillManager] Destroyed");
        }
    }

    // ========================================================================
    // Configuration and Gating
    // ========================================================================

    /**
     * @brief Check if COSMA should be enabled for given sequence length
     *
     * @param seq_len Sequence length to process
     * @return true if COSMA should be used, false for OpenBLAS fallback
     *
     * @note Gating logic: requires multi-rank MPI + seq_len >= threshold + not disabled
     */
    bool CosmaPrefillManager::enabled_for(int seq_len) const
    {
        const auto &env = debugEnv();

        // Explicit disable via environment
        if (env.cosma.disable)
        {
            return false;
        }

        // Need multi-rank for distributed operations
        if (world_size_ <= 1)
        {
            return false;
        }

        // Sequence length threshold
        if (seq_len < threshold_)
        {
            return false;
        }

        // Force enable if requested (API or env)
        if (force_cosma_)
        {
            return true;
        }

        return true;
    }

    // ========================================================================
    // Matrix Allocation with Destination-Local Population
    // ========================================================================

    /**
     * @brief Allocate a COSMA-distributed matrix
     *
     * @param label Matrix role ('A', 'B', 'C') for debugging
     * @param m Number of rows
     * @param n Number of columns
     * @param strat Distribution strategy to use
     * @param zero Whether to zero-initialize
     * @return CosmaView Wrapper around allocated COSMA matrix
     *
     * @note Tracks allocation for memory budget management
     * @note All matrices use 'A' label for COSMA (row-major from our perspective)
     */
    CosmaView CosmaPrefillManager::allocate_matrix(
        char label, int m, int n,
        const cosma::Strategy &strat,
        bool zero)
    {
        if (m <= 0 || n <= 0)
        {
            LOG_ERROR("[allocate_matrix] Invalid dimensions: " << m << "x" << n);
            return CosmaView{};
        }

        // Check memory budget
        size_t bytes_needed = static_cast<size_t>(m) * n * sizeof(cosma_scalar_t);
        if (!memory_budget_allows(bytes_needed))
        {
            LOG_WARN("[allocate_matrix] Memory budget exceeded for " << m << "x" << n);
            stats_.allocations_denied.fetch_add(1, std::memory_order_relaxed);
            return CosmaView{};
        }

        // Allocate COSMA matrix (always use 'A' for row-major)
        auto mat = std::make_shared<cosma::CosmaMatrix<cosma_scalar_t>>(
            label, strat, rank_);

        // Zero-initialize if requested
        if (zero)
        {
            cosma_scalar_t *ptr = mat->matrix_pointer();
            size_t local_size = mat->matrix_size();
            std::fill(ptr, ptr + local_size, cosma_scalar_t{0});
        }

        // Track allocation
        stats_.allocations_tracked.fetch_add(1, std::memory_order_relaxed);
        stats_.current_resident_bytes.fetch_add(bytes_needed, std::memory_order_relaxed);

        // Update peak
        long long current = stats_.current_resident_bytes.load(std::memory_order_relaxed);
        long long peak = stats_.peak_resident_bytes.load(std::memory_order_relaxed);
        while (current > peak)
        {
            if (stats_.peak_resident_bytes.compare_exchange_weak(
                    peak, current, std::memory_order_relaxed))
            {
                break;
            }
        }

        CosmaView view;
        view.mat = mat;
        view.global_rows = m;
        view.global_cols = n;
        view.label = label;
        view.strategy = &strat;

        return view;
    }

    /**
     * @brief Populate COSMA matrix using destination-local pattern (COSMA best practice)
     *
     * Iterates over local elements and uses global_coordinates() to determine
     * which source element to copy. This is O(local_n) per rank with exact mapping.
     *
     * @param dst COSMA matrix to populate
     * @param src_row_major Source data in row-major layout
     * @param rows Number of rows
     * @param cols Number of columns
     */
    void CosmaPrefillManager::scatter_row_major_dest_local(
        CosmaView &dst,
        const float *src_row_major,
        int rows,
        int cols)
    {
        if (!dst.mat || !src_row_major)
        {
            LOG_ERROR("[scatter_row_major] Invalid inputs");
            return;
        }

        cosma_scalar_t *dst_local = dst.mat->matrix_pointer();
        size_t local_size = dst.mat->matrix_size();

        // Destination-local population (COSMA best practice)
        for (size_t li = 0; li < local_size; ++li)
        {
            auto gc = dst.mat->global_coordinates(li);
            int gi = gc.first;  // global row
            int gj = gc.second; // global column

            // Bounds check and populate
            if (gi >= 0 && gi < rows && gj >= 0 && gj < cols)
            {
                // Convert float→double (cosma_scalar_t)
                dst_local[li] = static_cast<cosma_scalar_t>(
                    src_row_major[gi * cols + gj]);
            }
            else
            {
                dst_local[li] = cosma_scalar_t{0};
            }
        }

        stats_.bytes_converted_activations.fetch_add(
            rows * cols * sizeof(float),
            std::memory_order_relaxed);
    }

    // ========================================================================
    // Activation Conversion
    // ========================================================================

    /**
     * @brief Convert row-major activation matrix to COSMA distributed layout
     *
     * @param row_major Source activation data (float, row-major)
     * @param m Number of rows (sequence length)
     * @param k Number of columns (hidden dimension)
     * @return CosmaView Distributed COSMA matrix
     *
     * @note Uses unified strategy for consistency with matmul
     * @note Performs destination-local population
     */
    CosmaView CosmaPrefillManager::convert_activation_in(
        const float *row_major,
        int m,
        int k)
    {
        if (!row_major || m <= 0 || k <= 0)
        {
            LOG_ERROR("[convert_activation_in] Invalid inputs: m=" << m << ", k=" << k);
            return CosmaView{};
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Get unified strategy
        const auto &strat = strategy_cache_.get(m, k, k, world_size_);

        // Allocate matrix
        CosmaView view = allocate_matrix('A', m, k, strat, false);
        if (!view.mat)
        {
            return CosmaView{};
        }

        // Populate using destination-local pattern
        scatter_row_major_dest_local(view, row_major, m, k);

        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        stats_.us_convert_activation.fetch_add(us, std::memory_order_relaxed);

        return view;
    }

    // ========================================================================
    // Weight Loading
    // ========================================================================

    /**
     * @brief Load weight matrix into COSMA distributed layout
     *
     * @param desc Weight descriptor (dimensions, pointer, quantization info)
     * @return CosmaWeightHandle Handle to loaded weight
     *
     * @note Currently supports float32 weights only
     * @note Quantized weight support deferred to Phase 2
     */
    CosmaWeightHandle CosmaPrefillManager::load_weight(const WeightDescriptor &desc)
    {
        if (!desc.base_ptr || desc.rows <= 0 || desc.cols <= 0)
        {
            LOG_ERROR("[load_weight] Invalid descriptor: "
                      << desc.rows << "x" << desc.cols);
            return CosmaWeightHandle{};
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Get unified strategy
        const auto &strat = strategy_cache_.get(desc.rows, desc.cols, desc.cols, world_size_);

        // Allocate matrix
        CosmaView view = allocate_matrix('B', desc.rows, desc.cols, strat, false);
        if (!view.mat)
        {
            return CosmaWeightHandle{};
        }

        // Populate from source
        if (desc.quant_type == 0)
        {
            // Float32 weights
            const float *src = static_cast<const float *>(desc.base_ptr);
            scatter_row_major_dest_local(view, src, desc.rows, desc.cols);
        }
        else
        {
            LOG_ERROR("[load_weight] Quantized weights not yet supported in refactored version");
            return CosmaWeightHandle{};
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        stats_.us_stream_weights.fetch_add(us, std::memory_order_relaxed);
        stats_.bytes_streamed_weights.fetch_add(
            desc.rows * desc.cols * sizeof(float),
            std::memory_order_relaxed);

        CosmaWeightHandle handle;
        handle.view = std::move(view);
        handle.desc = desc;

        return handle;
    }

    void CosmaPrefillManager::release_weight(CosmaWeightHandle &&handle)
    {
        // Automatic cleanup via RAII
        if (handle.view.mat)
        {
            size_t bytes = handle.desc.rows * handle.desc.cols * sizeof(cosma_scalar_t);
            stats_.current_resident_bytes.fetch_sub(bytes, std::memory_order_relaxed);
        }
        // Let handle go out of scope for automatic cleanup
    }

    // ========================================================================
    // Matrix Multiplication
    // ========================================================================

    /**
     * @brief Perform distributed matrix multiplication using COSMA
     *
     * Computes: C = alpha * A * W + beta * C
     *
     * COSMA Memory Allocation:
     * ========================
     * This function allocates a COSMA matrix for the result (C) from the global LIFO
     * memory pool. The caller is responsible for ensuring LIFO-safe cleanup:
     *
     *   - If returning C to caller: Must reconstruct and destroy before other allocations
     *   - If creating multiple results: Must destroy in reverse allocation order
     *   - See fused_rmsnorm_qkv() for example of safe multi-result pattern
     *
     * Memory Management Contract:
     *   - Input: A and W.view.mat must be valid COSMA matrices
     *   - Output: C.mat allocated from COSMA memory pool (adds to pool_size_)
     *   - Cleanup: Caller must ensure C freed in LIFO order (see examples)
     *
     * Release Chain:
     *   C.release_chain contains A.mat and W.view.mat to extend their lifetimes
     *   until C is destroyed. This prevents premature deallocation of operands.
     *   Note: Clearing release_chain does NOT free COSMA buffers immediately.
     *         Must explicitly call .mat.reset() to trigger LIFO free.
     *
     * Example - Single Result (SAFE):
     *   auto C = matmul(A, W, m, k, n);
     *   // Use C...
     *   // C.mat destroyed on scope exit (LIFO maintained)
     *
     * Example - Multiple Results (UNSAFE without reconstruction):
     *   auto Q = matmul(A, Wq, m, k, n);  // Allocates Q buffer
     *   auto K = matmul(A, Wk, m, k, n);  // Allocates K buffer
     *   auto V = matmul(A, Wv, m, k, n);  // Allocates V buffer
     *   // DANGER: Q, K, V all alive - destruction order undefined!
     *   // Solution: Reconstruct and destroy immediately (see fused_rmsnorm_qkv)
     *
     * @param A Activation matrix (m×k)
     * @param W Weight handle containing matrix (k×n)
     * @param m Number of rows in A and C
     * @param k Inner dimension
     * @param n Number of columns in C
     * @param transposeW Transpose flag (currently unused, assumes no transpose)
     * @param alpha Scaling factor for product (default 1.0)
     * @param beta Scaling factor for existing C (default 0.0)
     * @return CosmaView Result matrix C (m×n) with .mat allocated from COSMA pool
     *
     * @note Uses MPI barriers before/after COSMA multiply to prevent hangs
     * @note Tracks timing and operation count in statistics
     * @note Result C.release_chain contains [A.mat, W.view.mat] for lifetime extension
     *
     * @see fused_rmsnorm_qkv() for example of LIFO-safe multi-result pattern
     * @see cosma.instructions.md for comprehensive LIFO management guidelines
     *
     * @author David Sanftenberg
     */
    CosmaView CosmaPrefillManager::matmul(
        const CosmaView &A,
        const CosmaWeightHandle &W,
        int m, int k, int n,
        bool transposeW,
        float alpha,
        float beta)
    {
        if (!A.mat || !W.view.mat)
        {
            LOG_ERROR("[matmul] Invalid input matrices");
            return CosmaView{};
        }

        // Dimension validation: W must be (k x n)
        // (Global rows correspond to k, global cols correspond to n)
        if (W.view.global_rows != k || W.view.global_cols != n)
        {
            LOG_ERROR("[matmul][dim-mismatch] Expected W " << k << "x" << n
                                                           << " but descriptor has " << W.view.global_rows << "x" << W.view.global_cols
                                                           << " (id=" << W.desc.id << ")");
            // Fail fast to avoid silent memory corruption.
            return CosmaView{};
        }

        // Debug: allow disabling release_chain retention to isolate double-free source.
        static int disable_chain = []()
        {
            const char *v = std::getenv("LLAMINAR_COSMA_DISABLE_RELEASE_CHAIN");
            int val = (v && *v && std::string(v) != "0") ? 1 : 0;
            // Log on first call only
            static std::once_flag log_once;
            std::call_once(log_once, [v, val]()
                           { LOG_INFO("[COSMA] Startup: LLAMINAR_COSMA_DISABLE_RELEASE_CHAIN="
                                      << (v ? v : "<unset>") << " -> disable_chain=" << val); });
            return val;
        }();
        static int dtor_trace = []()
        {
            const char *v = std::getenv("LLAMINAR_COSMA_DTOR_TRACE");
            return (v && *v && std::string(v) != "0") ? 1 : 0;
        }();

        auto start = std::chrono::high_resolution_clock::now();

        // ========================================================================
        // Dimension Validation - Prevent Silent Corruption
        // ========================================================================
        // Validate that weight matrix dimensions match GEMM requirements:
        //   C[m,n] = A[m,k] * B[k,n]
        //   Weight (B) must have: rows=k (inner dimension), cols=n (output dimension)
        //
        if (W.view.global_rows != k || W.view.global_cols != n)
        {
            LOG_ERROR("[matmul] Weight dimension mismatch!");
            LOG_ERROR("  Expected weight: [" << k << ", " << n << "] (k×n for GEMM)");
            LOG_ERROR("  Actual weight:   [" << W.view.global_rows << ", " << W.view.global_cols << "]");
            LOG_ERROR("  GEMM shape: C[" << m << "," << n << "] = A[" << m << "," << k << "] * W[" << k << "," << n << "]");
            LOG_ERROR("  This indicates WeightDescriptor was constructed incorrectly!");
            LOG_ERROR("  Should be: WeightDescriptor{id, in_features, out_features, ...}");
            return CosmaView{};
        }

        // Get unified strategy for result
        const auto &strat = strategy_cache_.get(m, n, k, world_size_);

        // ========================================================================
        // COSMA Buffer Allocation - LIFO Memory Pool
        // ========================================================================
        // Allocate result matrix C from COSMA's global LIFO memory pool.
        // This increments pool_size_ and returns a pointer at pool_[pool_size_].
        //
        // CRITICAL: This buffer MUST be freed in LIFO order (last allocated, first freed).
        // If caller creates multiple results (e.g., Q, K, V), they MUST:
        //   1. Reconstruct to host memory immediately after creation
        //   2. Destroy in reverse allocation order (V→K→Q)
        //   3. See fused_rmsnorm_qkv() for reference implementation
        //
        // Failure to maintain LIFO will cause:
        //   Assertion failed: pool_.data() + pool_size_ == ptr
        //   in memory_pool.cpp::free_buffer()
        //
        CosmaView C = allocate_matrix('C', m, n, strat, true);
        if (!C.mat)
        {
            return CosmaView{};
        }

        // Perform distributed multiply with barriers (prevent hangs)
        safe_barrier();

        cosma::multiply(
            *A.mat,      // A matrix
            *W.view.mat, // B matrix (weight)
            *C.mat,      // C matrix (result)
            strat,
            MPI_COMM_WORLD,
            static_cast<cosma_scalar_t>(alpha),
            static_cast<cosma_scalar_t>(beta));

        safe_barrier();

        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        stats_.matmul_invocations.fetch_add(1, std::memory_order_relaxed);
        stats_.us_matmul.fetch_add(us, std::memory_order_relaxed);

        if (rank_ == 0)
        {
            double ms = us / 1000.0;
            double gflops = (2.0 * m * n * k) / (us * 1000.0);
            LOG_DEBUG("[matmul] " << m << "×" << k << " * " << k << "×" << n
                                  << " = " << m << "×" << n
                                  << " | " << ms << " ms, " << gflops << " GFLOPS");
        }

        // ========================================================================
        // Release Chain - Operand Lifetime Extension
        // ========================================================================
        // The release_chain vector extends operand lifetimes until result C is destroyed.
        // This prevents premature deallocation when operands go out of scope.
        //
        // IMPORTANT: Release chain ordering affects C++ destructor sequence but does NOT
        // directly control COSMA buffer deallocation timing. To free COSMA buffers, must
        // explicitly call .mat.reset() - see capture_host_owned in fused_rmsnorm_qkv().
        //
        // Ordering rationale (must match allocation order for correct ref counting):
        //   - A allocated/loaded first → push first
        //   - W allocated/loaded second → push second
        //   - C allocated last (this function)
        //
        // When C.mat destroyed, release_chain also destroyed, decrementing A and W refs.
        // This ordering ensures A and W outlive C, preventing use-after-free.
        //
        // NOTE: In multi-result scenarios (Q/K/V), clearing release_chain does NOT
        // immediately free COSMA buffers. Must use explicit reconstruction + reset pattern.
        //
        if (!disable_chain)
        {
            C.release_chain.push_back(A.mat);      // Earliest allocation first
            C.release_chain.push_back(W.view.mat); // Weight second
        }
        else if (rank_ == 0)
        {
            LOG_WARN("[matmul] release_chain disabled via LLAMINAR_COSMA_DISABLE_RELEASE_CHAIN");
        }

        if (dtor_trace && rank_ == 0)
        {
            LOG_DEBUG("[matmul][trace] alloc C=" << (void *)C.mat.get()
                                                 << " A.use=" << (A.mat ? A.mat.use_count() : 0)
                                                 << " W.use=" << (W.view.mat ? W.view.mat.use_count() : 0));
        }

        return C;
    }

    // ========================================================================
    // Result Reconstruction (Gather from Distributed Layout)
    // ========================================================================

    /**
     * @brief Reconstruct distributed COSMA matrix to row-major host buffer
     *
     * Gathers distributed matrix data from all ranks using Allreduce.
     *
     * @param src COSMA distributed matrix
     * @param dst Output buffer (row-major, pre-allocated)
     * @param normalize Unused in minimal implementation
     *
     * @note Output buffer must be pre-allocated with rows×cols×sizeof(float) bytes
     * @note Performs double→float conversion at boundary
     */
    void CosmaPrefillManager::reconstruct_matrix(
        const CosmaView &src,
        float *dst,
        bool normalize) const
    {
        if (!src.mat || !dst)
        {
            LOG_ERROR("[reconstruct_matrix] Invalid inputs");
            return;
        }

        int rows = src.global_rows;
        int cols = src.global_cols;

        // Temporary double buffer for gather
        std::vector<cosma_scalar_t> temp(rows * cols, cosma_scalar_t{0});

        // Gather local elements to global positions
        cosma_scalar_t *local_ptr = src.mat->matrix_pointer();
        size_t local_size = src.mat->matrix_size();

        for (size_t li = 0; li < local_size; ++li)
        {
            auto gc = src.mat->global_coordinates(li);
            int gi = gc.first;
            int gj = gc.second;

            if (gi >= 0 && gi < rows && gj >= 0 && gj < cols)
            {
                temp[gi * cols + gj] = local_ptr[li];
            }
        }

        // Allreduce to combine from all ranks
        if (world_size_ > 1)
        {
            std::vector<cosma_scalar_t> global(rows * cols);
            MPI_Allreduce(
                temp.data(),
                global.data(),
                rows * cols,
                MPI_DOUBLE, // cosma_scalar_t is double
                MPI_SUM,
                MPI_COMM_WORLD);
            temp = std::move(global);
        }

        // Convert double→float and write to output
        for (int i = 0; i < rows * cols; ++i)
        {
            dst[i] = static_cast<float>(temp[i]);
        }
    }

    void CosmaPrefillManager::to_row_major(
        const CosmaView &src,
        float *dst,
        bool force_normalize) const
    {
        // If already reconstructed to host memory, just copy
        if (src.host_owned && src.original_row_major)
        {
            size_t elements = static_cast<size_t>(src.global_rows) * src.global_cols;
            std::memcpy(dst, src.original_row_major, elements * sizeof(float));
            return;
        }

        // Otherwise, reconstruct from distributed matrix
        reconstruct_matrix(src, dst, force_normalize);
    }

    // ========================================================================
    // Fused RMSNorm + QKV Projections
    // ========================================================================

    /**
     * @brief Fused RMSNorm + Q/K/V projections using proven primitives + COSMA
     *
     * Implementation flow:
     * 1. RMSNorm using proven rmsnorm_t5_forward() (matches HuggingFace)
     * 2. Load Q/K/V weights into COSMA distributed layout
     * 3. Convert normalized activations to COSMA layout
     * 4. Perform distributed Q/K/V projections via COSMA matmul
     * 5. CRITICAL: Reconstruct and destroy matrices in LIFO order (see below)
     * 6. Return results with host-owned buffers
     *
     * COSMA LIFO Memory Management:
     * ==============================
     * COSMA uses a global strict LIFO (stack-based) bump allocator. Buffers MUST
     * be freed in exact reverse order of allocation. Violating this causes:
     *   Assertion failed: pool_.data() + pool_size_ == ptr
     *
     * Problem: Creating Q, K, V matrices simultaneously creates overlapping lifetimes:
     *   - Allocation order: Q → K → V
     *   - C++ destructor order: undefined (implementation-dependent)
     *   - Result: LIFO violation → crash
     *
     * Solution: Immediately reconstruct to host memory and destroy in LIFO order:
     *   1. Execute Q/K/V matmuls (temporary overlap acceptable)
     *   2. Reconstruct V to host buffer, destroy V.mat (free buffer)
     *   3. Reconstruct K to host buffer, destroy K.mat (free buffer)
     *   4. Reconstruct Q to host buffer, destroy Q.mat (free buffer)
     *   5. Reconstruct norm to host buffer, destroy norm.mat (free buffer)
     *
     * This pattern ensures:
     *   - LIFO invariant maintained (V freed before K, K before Q)
     *   - No overlapping COSMA buffer lifetimes
     *   - Results available as host_owned std::shared_ptr (automatic cleanup)
     *
     * Performance Impact:
     *   - Reconstruction overhead: ~1-4ms for multi-rank gather
     *   - Acceptable trade-off for correctness
     *   - Required anyway for downstream operations
     *
     * @param activation_row_major Input activations (seq_len × hidden_size)
     * @param gamma RMSNorm weights (hidden_size)
     * @param wq Query projection weight descriptor
     * @param wk Key projection weight descriptor
     * @param wv Value projection weight descriptor
     * @param seq_len Sequence length
     * @param hidden_size Hidden dimension
     * @param eps RMSNorm epsilon
     * @param softmax_scale Unused (for API compatibility)
     * @param transpose_k Unused (for API compatibility)
     * @return FusedRmsnormQkvResult Result structure with Q/K/V host_owned buffers
     *
     * @note All returned CosmaView objects have .mat=nullptr and .host_owned populated
     * @note Release chains cleared - no lingering COSMA resource dependencies
     * @note See cosma.instructions.md for detailed LIFO management guidelines
     *
     * @author David Sanftenberg
     */
    FusedRmsnormQkvResult CosmaPrefillManager::fused_rmsnorm_qkv(
        const float *activation_row_major,
        const float *gamma,
        const WeightDescriptor &wq,
        const WeightDescriptor &wk,
        const WeightDescriptor &wv,
        int seq_len,
        int hidden_size,
        float eps,
        float softmax_scale,
        bool transpose_k)
    {
        auto start = std::chrono::high_resolution_clock::now();

        FusedRmsnormQkvResult result;

        // Step 1: RMSNorm using proven primitive
        auto normalized = std::make_shared<std::vector<float>>(seq_len * hidden_size);

        llaminar::kernels::rmsnorm_t5_forward(
            activation_row_major,
            gamma,
            normalized->data(),
            seq_len,
            hidden_size,
            eps,
            true // use_parallel
        );

        // Step 2: Load weights
        auto wq_handle = load_weight(wq);
        auto wk_handle = load_weight(wk);
        auto wv_handle = load_weight(wv);

        if (!wq_handle.view.mat || !wk_handle.view.mat || !wv_handle.view.mat)
        {
            LOG_ERROR("[fused_rmsnorm_qkv] Failed to load weights");
            return result;
        }

        // Step 3: Convert normalized activations to COSMA layout
        CosmaView norm_view = convert_activation_in(
            normalized->data(),
            seq_len,
            hidden_size);

        if (!norm_view.mat)
        {
            LOG_ERROR("[fused_rmsnorm_qkv] Failed to convert activations");
            return result;
        }

        // Step 4: Distributed Q/K/V projections
        // CRITICAL: To maintain COSMA LIFO invariants, we must reconstruct and destroy
        // the COSMA matrices immediately after each operation. This matches the legacy
        // implementation's capture_host_owned pattern which prevents overlapping buffer
        // lifetimes in COSMA's strict LIFO memory pool.
        //
        // Background: COSMA uses a global bump allocator with strict LIFO discipline.
        // Creating Q, K, V matrices simultaneously violates this because their destruction
        // order is undefined. The solution is to:
        //   1. Execute the matmuls (temporary overlap acceptable during computation)
        //   2. Immediately reconstruct each result to host memory
        //   3. Destroy COSMA matrix objects in reverse allocation order (V→K→Q)
        //   4. Return views with host_owned buffers and mat=nullptr
        //
        // See function-level documentation and cosma.instructions.md for full details.

        CosmaView q_view = matmul(norm_view, wq_handle, seq_len, hidden_size, wq.cols, false);
        if (!q_view.mat)
        {
            LOG_ERROR("[fused_rmsnorm_qkv] Q matmul failed");
            return result;
        }

        CosmaView k_view = matmul(norm_view, wk_handle, seq_len, hidden_size, wk.cols, false);
        if (!k_view.mat)
        {
            LOG_ERROR("[fused_rmsnorm_qkv] K matmul failed");
            return result;
        }

        CosmaView v_view = matmul(norm_view, wv_handle, seq_len, hidden_size, wv.cols, false);
        if (!v_view.mat)
        {
            LOG_ERROR("[fused_rmsnorm_qkv] V matmul failed");
            return result;
        }

        // ========================================================================
        // COSMA LIFO Memory Management - Immediate Reconstruction and Destruction
        // ========================================================================
        //
        // This lambda implements the critical LIFO-safe cleanup pattern:
        //   1. Reconstruct distributed matrix to host memory via MPI Allreduce
        //   2. Destroy COSMA matrix object to free buffer in LIFO order
        //   3. Clear release chains to allow dependency cleanup
        //
        // WHY THIS IS NECESSARY:
        // - COSMA allocator: pool_[offset] → pool_[offset+size] (bump pointer)
        // - Free requirement: assert(pool_.data() + pool_size_ == ptr)
        // - Violation: "free mismatch size=N expected=0xA got=0xB"
        //
        // ALLOCATION SEQUENCE (when Q/K/V created):
        //   Q.mat allocated → offset 0, size X
        //   K.mat allocated → offset X, size Y
        //   V.mat allocated → offset X+Y, size Z
        //   pool_size_ = X+Y+Z
        //
        // REQUIRED FREE SEQUENCE (must be exact reverse):
        //   free V.mat (pool_size_ X+Y+Z → X+Y) ✓
        //   free K.mat (pool_size_ X+Y → X)     ✓
        //   free Q.mat (pool_size_ X → 0)       ✓
        //
        // C++ destructor order is UNDEFINED, so we cannot rely on it.
        // We must explicitly reconstruct and destroy in LIFO order.
        //
        auto capture_host_owned = [&](CosmaView &view) -> bool
        {
            const size_t elements = static_cast<size_t>(view.global_rows) * view.global_cols;

            // Already host-owned, nothing to do
            if (!view.mat)
            {
                if (!view.host_owned && view.original_row_major && elements > 0)
                {
                    auto buf = std::make_shared<std::vector<float>>(elements, 0.f);
                    std::memcpy(buf->data(), view.original_row_major, elements * sizeof(float));
                    view.host_owned = buf;
                    view.original_row_major = buf->data();
                }
                return true;
            }

            // Step 1: Reconstruct distributed matrix to host memory
            // Uses MPI Allreduce to gather distributed data from all ranks
            // Converts double→float at boundary (COSMA internal type is double)
            auto buf = std::make_shared<std::vector<float>>(elements, 0.f);
            reconstruct_matrix(view, buf->data(), view.label != 'C');
            view.host_owned = buf;
            view.original_row_major = buf->data();

            // Step 2: CRITICAL - Destroy COSMA matrix to free buffer in LIFO order
            // This MUST happen for BOTH single-rank and multi-rank to prevent overlapping lifetimes.
            // The shared_ptr reset immediately decrements ref count, triggering CosmaMatrix destructor,
            // which calls memory_pool::free_buffer(), which enforces LIFO via assertion.
            view.mat.reset();

            // Step 3: Clear release chain to allow dependency cleanup
            // Release chain holds shared_ptr to operand matrices (A and W in C = A*W).
            // Clearing allows their ref counts to drop, enabling their eventual destruction.
            // Note: This doesn't affect LIFO for THIS matrix (already freed above).
            for (auto it = view.release_chain.rbegin(); it != view.release_chain.rend(); ++it)
            {
                it->reset();
            }
            view.release_chain.clear();
            return true;
        };

        // ========================================================================
        // LIFO Destruction Sequence - MUST BE IN REVERSE ALLOCATION ORDER
        // ========================================================================
        //
        // Allocation sequence (earlier in this function):
        //   norm_view allocated (Step 3)
        //   q_view allocated (Step 4.1)
        //   k_view allocated (Step 4.2)
        //   v_view allocated (Step 4.3)
        //
        // Required destruction sequence (LIFO - last allocated freed first):
        //   free v_view  (offset X+Y+Z → X+Y)
        //   free k_view  (offset X+Y → X)
        //   free q_view  (offset X → 0)
        //   free norm_view (varies based on strategy)
        //
        // DO NOT change this order or COSMA will crash with assertion failure!
        //
        capture_host_owned(v_view);    // Free V buffer (allocated last)
        capture_host_owned(k_view);    // Free K buffer
        capture_host_owned(q_view);    // Free Q buffer
        capture_host_owned(norm_view); // Free norm buffer (allocated first)

        if (rank_ == 0)
        {
            LOG_DEBUG("[fused_rmsnorm_qkv] Captured all outputs to host memory");

            // Diagnostic statistics for parity debugging
            static int enable_stats = []()
            {
                const char *v = std::getenv("LLAMINAR_COSMA_QKV_STATS");
                return (v && *v && std::string(v) != "0") ? 1 : 0;
            }();

            if (enable_stats && q_view.host_owned && k_view.host_owned && v_view.host_owned)
            {
                auto compute_stats = [](const float *data, size_t n, const char *name)
                {
                    if (!data || n == 0)
                        return;
                    float min_val = data[0], max_val = data[0];
                    double sum = 0.0;
                    for (size_t i = 0; i < n; ++i)
                    {
                        float v = data[i];
                        if (v < min_val)
                            min_val = v;
                        if (v > max_val)
                            max_val = v;
                        sum += v;
                    }
                    float mean = static_cast<float>(sum / n);
                    LOG_INFO("[QKV_STATS] " << name << ": min=" << min_val
                                            << ", max=" << max_val << ", mean=" << mean
                                            << ", elements=" << n);
                };

                compute_stats(q_view.host_owned->data(), q_view.host_owned->size(), "Q");
                compute_stats(k_view.host_owned->data(), k_view.host_owned->size(), "K");
                compute_stats(v_view.host_owned->data(), v_view.host_owned->size(), "V");
                compute_stats(norm_view.host_owned->data(), norm_view.host_owned->size(), "norm");
            }
        }

        // Populate result - keep activation and weight guards alive
        result.activation_guard = norm_view;
        result.normalized = norm_view;
        result.wq_guard = FusedRmsnormQkvResult::WeightGuard(this, std::move(wq_handle));
        result.wk_guard = FusedRmsnormQkvResult::WeightGuard(this, std::move(wk_handle));
        result.wv_guard = FusedRmsnormQkvResult::WeightGuard(this, std::move(wv_handle));
        result.q = std::move(q_view);
        result.k = std::move(k_view);
        result.v = std::move(v_view);

        // Statistics
        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        stats_.fused_rmsnorm_qkv_invocations.fetch_add(1, std::memory_order_relaxed);
        stats_.us_fused_rmsnorm_qkv.fetch_add(us, std::memory_order_relaxed);

        if (rank_ == 0)
        {
            double ms = us / 1000.0;
            LOG_INFO("[fused_rmsnorm_qkv] seq_len=" << seq_len
                                                    << ", hidden=" << hidden_size
                                                    << " | " << ms << " ms");
        }

        return result;
    }

    // ========================================================================
    // Memory Budget Management
    // ========================================================================

    bool CosmaPrefillManager::memory_budget_allows(size_t bytes_needed) const
    {
        long long current = stats_.current_resident_bytes.load(std::memory_order_relaxed);
        long long budget = max_resident_mb_ * 1024LL * 1024LL;

        if (current + static_cast<long long>(bytes_needed) > budget)
        {
            if (rank_ == 0)
            {
                LOG_WARN("[memory_budget] Allocation denied: current="
                         << (current / 1024 / 1024) << " MB + requested="
                         << (bytes_needed / 1024 / 1024) << " MB > budget="
                         << max_resident_mb_ << " MB");
            }
            return false;
        }

        return true;
    }

    // ========================================================================
    // Wrapper methods for backward compatibility
    // ========================================================================

    /**
     * @brief Wrapper for convert_activation_in - used by adaptive_matmul.h
     *
     * Note: The refactored version internally gets the strategy, so we ignore the passed strat.
     */
    CosmaView CosmaPrefillManager::convert_activation_operand(const float *row_major, int m, int k, const cosma::Strategy &strat)
    {
        (void)strat; // Ignored - refactored version gets strategy internally
        return convert_activation_in(row_major, m, k);
    }

    /**
     * @brief Wrapper for load_weight - used by adaptive_matmul.h
     *
     * Note: The refactored version internally gets the strategy, so we ignore the passed strat.
     */
    CosmaWeightHandle CosmaPrefillManager::load_weight_operand(const WeightDescriptor &desc, const cosma::Strategy &strat)
    {
        (void)strat; // Ignored - refactored version gets strategy internally
        return load_weight(desc);
    }

    /**
     * @brief Debug validation helper - no-op in refactored version
     *
     * The original implementation had complex validation infrastructure.
     * In the minimal refactored version, we skip this diagnostic.
     */
    void CosmaPrefillManager::debug_compare_original(const CosmaView &src, int rows, int cols, const float *original) const
    {
        // No-op in refactored version - diagnostic infrastructure removed for simplicity
        // If needed, can add lightweight validation in Phase 2
        (void)src;
        (void)rows;
        (void)cols;
        (void)original;
    }

    // ========================================================================
    // Statistics Management
    // ========================================================================

    void CosmaPrefillManager::reset_stats()
    {
        stats_.single_rank_calls.store(0, std::memory_order_relaxed);
        stats_.fast_path_calls.store(0, std::memory_order_relaxed);
        stats_.cosma_path_calls.store(0, std::memory_order_relaxed);
        stats_.bytes_streamed_weights.store(0, std::memory_order_relaxed);
        stats_.bytes_converted_activations.store(0, std::memory_order_relaxed);
        stats_.matmul_invocations.store(0, std::memory_order_relaxed);
        stats_.us_stream_weights.store(0, std::memory_order_relaxed);
        stats_.us_convert_activation.store(0, std::memory_order_relaxed);
        stats_.us_matmul.store(0, std::memory_order_relaxed);
        stats_.fused_rmsnorm_qkv_invocations.store(0, std::memory_order_relaxed);
        stats_.us_fused_rmsnorm_qkv.store(0, std::memory_order_relaxed);

        // Don't reset current_resident_bytes or peak - those track actual state
    }

} // namespace llaminar
