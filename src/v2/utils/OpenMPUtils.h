/**
 * @file OpenMPUtils.h
 * @brief OpenMP utility macros for nested-safe parallel regions
 * @author David Sanftenberg
 *
 * Provides macros to create OpenMP parallel regions that are safe to call
 * from within existing parallel regions. This enables "layer-level fusion"
 * where an outer parallel region can encompass multiple kernel calls,
 * eliminating thread fork/join overhead between operations.
 *
 * The key insight is that OpenMP worksharing constructs (#pragma omp for)
 * require an active parallel region, but we don't want to create a new
 * parallel region if we're already inside one (nested parallelism has
 * significant overhead).
 *
 * Usage Pattern:
 * @code
 *   // Define work as a lambda with worksharing constructs
 *   auto do_work = [&]() {
 *       #pragma omp for schedule(static) nowait
 *       for (size_t i = 0; i < n; ++i) {
 *           result[i] = compute(input[i]);
 *       }
 *   };
 *
 *   // Execute: creates parallel region only if not already in one
 *   OMP_WORKSHARE_REGION(do_work);
 * @endcode
 *
 * This allows kernels to work correctly both:
 * 1. Standalone (creates own parallel region)
 * 2. Inside a layer-level parallel region (uses existing threads)
 */

#pragma once

#include <omp.h>

namespace llaminar::v2
{

/**
 * @brief Execute a worksharing lambda, creating a parallel region only if needed.
 *
 * If already inside a parallel region (omp_in_parallel() returns true),
 * the lambda is called directly. Otherwise, a new parallel region is created.
 *
 * @param work_fn A callable (typically a lambda) containing OpenMP worksharing
 *                constructs like #pragma omp for. Must be callable with no arguments.
 *
 * Example:
 * @code
 *   auto work = [&]() {
 *       #pragma omp for schedule(static)
 *       for (int i = 0; i < n; ++i) { ... }
 *   };
 *   OMP_WORKSHARE_REGION(work);
 * @endcode
 */
#define OMP_WORKSHARE_REGION(work_fn)              \
    do                                             \
    {                                              \
        if (omp_in_parallel())                     \
        {                                          \
            work_fn();                             \
        }                                          \
        else                                       \
        {                                          \
            _Pragma("omp parallel") { work_fn(); } \
        }                                          \
    } while (0)

/**
 * @brief Execute a worksharing lambda with a barrier, creating parallel region if needed.
 *
 * Same as OMP_WORKSHARE_REGION but adds an implicit barrier after the work.
 * Use this when subsequent code depends on all threads completing their work.
 *
 * @param work_fn A callable containing OpenMP worksharing constructs.
 *
 * Example:
 * @code
 *   auto work = [&]() {
 *       #pragma omp for schedule(static) nowait
 *       for (int i = 0; i < n; ++i) { ... }
 *   };
 *   OMP_WORKSHARE_REGION_SYNC(work);  // Barrier after worksharing
 * @endcode
 */
#define OMP_WORKSHARE_REGION_SYNC(work_fn)         \
    do                                             \
    {                                              \
        if (omp_in_parallel())                     \
        {                                          \
            work_fn();                             \
            _Pragma("omp barrier")                 \
        }                                          \
        else                                       \
        {                                          \
            _Pragma("omp parallel") { work_fn(); } \
        }                                          \
    } while (0)

/**
 * @brief Execute a worksharing lambda conditionally, with nested-safety.
 *
 * When `condition` is true and we're not already in a parallel region,
 * creates a new parallel region. When `condition` is true and we're already
 * in a parallel region, uses worksharing on existing threads.
 * When `condition` is false, executes serially on a single thread.
 *
 * This handles the common pattern of:
 *   #pragma omp parallel for if (should_parallelize)
 *
 * @param work_fn A callable containing OpenMP worksharing constructs.
 * @param condition Boolean expression determining whether to parallelize.
 *
 * Example:
 * @code
 *   bool parallelize_heads = (n_heads >= 4);
 *   auto work = [&]() {
 *       #pragma omp for schedule(static)
 *       for (int h = 0; h < n_heads; ++h) { ... }
 *   };
 *   OMP_WORKSHARE_REGION_IF(work, parallelize_heads);
 * @endcode
 */
#define OMP_WORKSHARE_REGION_IF(work_fn, condition)                            \
    do                                                                         \
    {                                                                          \
        if (condition)                                                         \
        {                                                                      \
            if (omp_in_parallel())                                             \
            {                                                                  \
                work_fn();                                                     \
            }                                                                  \
            else                                                               \
            {                                                                  \
                _Pragma("omp parallel") { work_fn(); }                         \
            }                                                                  \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            /* Serial: use 1-thread parallel so #pragma omp for still works */ \
            _Pragma("omp parallel num_threads(1)") { work_fn(); }              \
        }                                                                      \
    } while (0)

/**
 * @brief Execute a worksharing lambda with collapse(2) scheduling, conditionally.
 *
 * Similar to OMP_WORKSHARE_REGION_IF but the lambda should use:
 *   #pragma omp for collapse(2) schedule(static)
 * instead of just #pragma omp for.
 *
 * @param work_fn A callable containing #pragma omp for collapse(2) worksharing.
 * @param condition Boolean expression determining whether to parallelize.
 */
#define OMP_WORKSHARE_COLLAPSE2_IF(work_fn, condition) \
    OMP_WORKSHARE_REGION_IF(work_fn, condition)

/**
 * @brief Execute a worksharing lambda with collapse(3) scheduling, conditionally.
 *
 * Similar to OMP_WORKSHARE_REGION_IF but the lambda should use:
 *   #pragma omp for collapse(3) schedule(static)
 * instead of just #pragma omp for.
 *
 * @param work_fn A callable containing #pragma omp for collapse(3) worksharing.
 * @param condition Boolean expression determining whether to parallelize.
 */
#define OMP_WORKSHARE_COLLAPSE3_IF(work_fn, condition) \
    OMP_WORKSHARE_REGION_IF(work_fn, condition)

    /**
     * @brief Check if currently executing within an OpenMP parallel region.
     *
     * Thin wrapper around omp_in_parallel() for consistency and documentation.
     *
     * @return true if inside a parallel region, false otherwise.
     */
    inline bool isInParallelRegion() { return omp_in_parallel() != 0; }

    /**
     * @brief Get the current thread ID within a parallel region.
     *
     * Thin wrapper around omp_get_thread_num() for consistency.
     *
     * @return Thread ID (0 to num_threads-1), or 0 if outside parallel region.
     */
    inline int getThreadId() { return omp_get_thread_num(); }

    /**
     * @brief Get the number of threads in the current parallel region.
     *
     * Thin wrapper around omp_get_num_threads() for consistency.
     *
     * @return Number of threads, or 1 if outside parallel region.
     */
    inline int getNumThreads() { return omp_get_num_threads(); }

/**
 * @brief Execute code in a single thread within a parallel region.
 *
 * If inside a parallel region, only one thread executes the code while
 * others wait at the implicit barrier. If outside a parallel region,
 * the code executes directly.
 *
 * This is essential for MPI collectives inside layer-level parallel regions:
 * - MPI collectives must be called by only one thread per rank
 * - Other threads wait at the barrier after the single block
 *
 * @param code_fn A callable to execute on a single thread.
 *
 * Example:
 * @code
 *   #pragma omp parallel
 *   {
 *       // Parallel GEMM (all threads participate)
 *       parallel_gemm_kernel();
 *
 *       // MPI collective (single thread, others wait)
 *       OMP_SINGLE([&]() {
 *           MPI_Allreduce(local_data, global_data, count, MPI_FLOAT, MPI_SUM, comm);
 *       });
 *
 *       // More parallel work
 *       another_parallel_kernel();
 *   }
 * @endcode
 */
#define OMP_SINGLE(code_fn)                      \
    do                                           \
    {                                            \
        if (omp_in_parallel())                   \
        {                                        \
            _Pragma("omp single") { code_fn(); } \
        }                                        \
        else                                     \
        {                                        \
            code_fn();                           \
        }                                        \
    } while (0)

/**
 * @brief Execute code in a single thread without barrier (nowait).
 *
 * Same as OMP_SINGLE but without the implicit barrier after.
 * Use when subsequent code doesn't depend on the single block completing.
 *
 * WARNING: Only use when you're sure the subsequent code doesn't read
 * data written in the single block.
 *
 * @param code_fn A callable to execute on a single thread.
 */
#define OMP_SINGLE_NOWAIT(code_fn)                      \
    do                                                  \
    {                                                   \
        if (omp_in_parallel())                          \
        {                                               \
            _Pragma("omp single nowait") { code_fn(); } \
        }                                               \
        else                                            \
        {                                               \
            code_fn();                                  \
        }                                               \
    } while (0)

/**
 * @brief Start a layer-level parallel region for fused kernel execution.
 *
 * This establishes an outer parallel region that persists across multiple
 * kernel calls within a transformer layer. Inner kernels detect this via
 * omp_in_parallel() and use worksharing instead of creating new regions.
 *
 * Benefits:
 * - Eliminates fork/join overhead between operations (~10-50µs per region)
 * - For a 24-layer model with ~20 ops/layer = 480 regions saved per token
 *
 * MPI operations inside should use OMP_SINGLE() to serialize collectives.
 *
 * @param body Code block to execute in the parallel region.
 *
 * Example:
 * @code
 *   OMP_LAYER_PARALLEL({
 *       // These kernels use existing threads (no fork/join overhead)
 *       rms_norm_kernel();      // Uses OMP_WORKSHARE_REGION internally
 *       gemm_kernel();          // Uses OMP_WORKSHARE_REGION internally
 *
 *       // MPI collective serialized to single thread
 *       OMP_SINGLE([&]() { mpi_ctx->allreduce(...); });
 *
 *       swiglu_kernel();        // More parallel work
 *   });
 * @endcode
 */
#define OMP_LAYER_PARALLEL(body)                                         \
    do                                                                   \
    {                                                                    \
        if (omp_in_parallel())                                           \
        {                                                                \
            /* Already in parallel region (nested call), just execute */ \
            body                                                         \
        }                                                                \
        else                                                             \
        {                                                                \
            _Pragma("omp parallel") { body }                             \
        }                                                                \
    } while (0)

    /**
     * @brief RAII guard for temporarily setting the number of threads.
     *
     * Useful when a specific operation needs a different thread count.
     * Restores the original setting on destruction.
     *
     * Example:
     * @code
     *   {
     *       ThreadCountGuard guard(4);  // Use 4 threads
     *       // ... parallel work with 4 threads ...
     *   }  // Original thread count restored
     * @endcode
     */
    class ThreadCountGuard
    {
    public:
        explicit ThreadCountGuard(int num_threads)
            : original_threads_(omp_get_max_threads())
        {
            omp_set_num_threads(num_threads);
        }

        ~ThreadCountGuard() { omp_set_num_threads(original_threads_); }

        // Non-copyable, non-movable
        ThreadCountGuard(const ThreadCountGuard &) = delete;
        ThreadCountGuard &operator=(const ThreadCountGuard &) = delete;
        ThreadCountGuard(ThreadCountGuard &&) = delete;
        ThreadCountGuard &operator=(ThreadCountGuard &&) = delete;

    private:
        int original_threads_;
    };

} // namespace llaminar::v2
