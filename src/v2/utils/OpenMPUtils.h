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

#include <atomic>
#include <cstdint>

namespace llaminar::v2
{

/**
 * @brief Process-wide count of every actual OpenMP team fork/join creation.
 *
 * Every real fork in the process observes itself exactly once through
 * @ref noteOpenMPTeamFork: the standalone fork branches of the workshare
 * macros below, and every raw `#pragma omp parallel` site of the runtime
 * inventory (a source-policy check holds that inventory exhaustively against
 * this authority). A fork is a fork whether the source spelled it as a macro
 * or a raw pragma, and the count must not depend on per-site discipline: new
 * raw sites must either note themselves or enter the named exemption list
 * under review. The NAMED exemptions (one-time work outside execution, each
 * owned by its file, checked by the fork-note source policy):
 * - app/MPIBootstrapPhase.cpp: one-time rank bootstrap affinity verification.
 * - kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h, CPUPreparedWeightWriter.cpp,
 *   kernels/cuda/gemm/CUDAWeightPacker.cpp, kernels/rocm/gemm/ROCmWeightPacker.cpp:
 *   one-time weight preparation and publication.
 * - loaders/MmapRegion.h, PreparedHalfWeights.h, WeightManager.cpp: one-time
 *   model load and mmap population.
 * - planning/PlanningMemoryBandwidthMeasurement.cpp: planning-time measurement.
 * - execution/moe/MoEExpertWeightService.cpp: expert GEMM engine preparation
 *   at graph build.
 * - utils/OpenMPUtils.h itself: the noting macro authority.
 *
 * The increment is a relaxed fetch-add of an inline variable plus the current
 * owner stack walk, so it adds no allocation, lock or acquisition to the fork
 * it describes (the fork itself costs microseconds). Publication into
 * PerfStats stays at owned boundaries (see OpenMPTeamForkEvidence.h): the
 * cached-forward decode step publishes its window cell as
 * "openmp_team_forks_decode_step" (broad-collection gate), and the per-stage
 * CPU timing seam publishes "openmp_team_forks_stage" cells under the
 * explicit stage-timing gate. Every counter is monotonic and shared by test
 * shards, so consumers read deltas between scopes and snapshots, never
 * absolutes.
 *
 * OWNERSHIP BOUNDARY, stated explicitly because thread-local state does not
 * propagate into a forked team: a fork is attributed to the owner cells on
 * the stack of the thread that EXECUTES the note. The thread that constructs
 * a scope attributes its own forks - including forks it executes while it is
 * also the master of some team. A team worker carries a fresh (empty)
 * thread-local stack, so a fork executed in worker context contributes to
 * the process total only and is never attributed to any owner cell. The
 * decode discipline that stages execute on one thread and workers never fork
 * is enforced by the retained-team authority (inLexicalWorkshare), not
 * assumed here; a regression pins the accounting consequence.
 */
inline std::atomic<std::uint64_t> openMPTeamForkCount{0};

/**
 * @brief One owner's fork accounting cell on the calling thread's owner stack.
 *
 * Attribution is owner-scoped, never process-global: concurrent executor
 * scopes (tensor-parallel endpoints, NUMA device graphs, benchmark harnesses)
 * count each other's forks only in the process total, never in their own
 * cells, and the global total stays complete. The cell participates in a
 * thread-local intrusive stack, so nested scopes on one thread each observe
 * the same fork: a stage cell inside a decode-window cell sees the fork and
 * so does the window. Threads that join a forked team as workers carry no
 * owner (fresh thread-local state) and contribute nothing but the total,
 * which matches the decode discipline that workers never fork.
 */
struct OpenMPTeamForkCell
{
    OpenMPTeamForkCell *next = nullptr; ///< Next enclosing owner on this thread.
    std::atomic<std::uint64_t> forks{0}; ///< Forks observed while this owner was current.

    OpenMPTeamForkCell() = default;
    OpenMPTeamForkCell(const OpenMPTeamForkCell &) = delete;
    OpenMPTeamForkCell &operator=(const OpenMPTeamForkCell &) = delete;
};

/** @brief The calling thread's innermost fork-accounting owner, or null outside every scope. */
inline thread_local OpenMPTeamForkCell *openMPTeamForkOwnerStack = nullptr;

/**
 * @brief Observe one real team fork/join about to happen on the calling thread.
 *
 * Contributes to the process total and to every owner cell currently pushed
 * on this thread's stack (typically one or two fetch-adds). A thread outside
 * every scope contributes only the total.
 */
inline void noteOpenMPTeamFork() noexcept
{
    openMPTeamForkCount.fetch_add(1, std::memory_order_relaxed);
    for (OpenMPTeamForkCell *owner = openMPTeamForkOwnerStack; owner != nullptr; owner = owner->next)
        owner->forks.fetch_add(1, std::memory_order_relaxed);
}

/**
 * @brief Attribute the fork/joins of one lexical scope to their owner.
 *
 * Construct the scope on the thread that executes the fork/join window (the
 * CPU executor thread; GPU stages fork no OpenMP region and construct no
 * scope). Concurrent scopes on different threads stay independent through
 * their thread-local stacks, and a scope destructed before its outer scopes
 * restores the stack exactly, so attribution survives nesting, overlap and
 * early returns without any serialization of production execution.
 */
class OpenMPTeamForkScope final
{
public:
    OpenMPTeamForkScope() : previous_(openMPTeamForkOwnerStack)
    {
        cell_.next = previous_;
        openMPTeamForkOwnerStack = &cell_;
    }
    ~OpenMPTeamForkScope() { openMPTeamForkOwnerStack = previous_; }
    OpenMPTeamForkScope(const OpenMPTeamForkScope &) = delete;
    OpenMPTeamForkScope &operator=(const OpenMPTeamForkScope &) = delete;

    /** @return The fork/join count observed while this owner was current. */
    [[nodiscard]] std::uint64_t forks() const noexcept
    {
        return cell_.forks.load(std::memory_order_relaxed);
    }

private:
    OpenMPTeamForkCell cell_;
    OpenMPTeamForkCell *previous_;
};

/** @return The process-wide fork/join total; consumers outside a scope use deltas between reads. */
[[nodiscard]] inline std::uint64_t openMPTeamForkSnapshot() noexcept
{
    return openMPTeamForkCount.load(std::memory_order_relaxed);
}

/** @brief Lexical placement of the current task relative to OpenMP teams. */
enum class TeamPlacement
{
    Standalone, ///< No enclosing parallel region.
    Retained,   ///< Inside an enclosing region that is serialized/inactive (num_threads(1) or if(false)).
    Active,     ///< Inside an active parallel region with real concurrency.
};
/** @brief Classify the enclosing OpenMP context through one canonical authority.
 * @return Active when the current task is enclosed by an ACTIVE parallel region
 * (exact OpenMP omp_in_parallel() meaning, even if the immediately enclosing
 * region is a nested-serialized one-thread team with an active ancestor);
 * Retained when any enclosing region exists but none is active (serialized or
 * if(false) regions without an active ancestor); Standalone otherwise.
 * @details Retained-team reuse is keyed to the LEXICAL omp_get_level() fact;
 * the Active label never claims the current team width exceeds one. Callers
 * that genuinely need active concurrency keep inActiveParallelRegion().
 */
[[nodiscard]] inline TeamPlacement teamPlacement() noexcept
{
    if (omp_in_parallel() != 0) return TeamPlacement::Active;
    return omp_get_level() > 0 ? TeamPlacement::Retained : TeamPlacement::Standalone;
}
/** @brief Whether an enclosing parallel region exists at all (active, serialized or inactive).
 * @details The canonical retained-team predicate: a `num_threads(1)` or
 * `if(false)` region is still a lexical team whose participants must reuse the
 * exact team, never fork an inner region.
 */
[[nodiscard]] inline bool inLexicalWorkshare() noexcept { return teamPlacement() != TeamPlacement::Standalone; }
/** @brief Whether the current task is enclosed by an active parallel region.
 * @details Exact OpenMP omp_in_parallel() meaning: an active ENCLOSING region
 * exists, which may be an ancestor while the current (nested-serialized) team
 * has exactly one thread. This does not imply current team width > 1.
 */
[[nodiscard]] inline bool inActiveParallelRegion() noexcept { return teamPlacement() == TeamPlacement::Active; }

/**
 * @brief Admit useful work or complete an empty retained-team invocation.
 * @param useful Whether immutable original publications admit any useful work.
 * @return The supplied decision, after empty retained members rendezvous.
 * @details Every member of an existing retained team (active, serialized or
 * inactive) calls this boundary. Empty operations need the same completion
 * edge as an omp-for implicit barrier: otherwise a fast member can publish the
 * next request before a late member has observed the current empty
 * publication. The barrier is lexical: inside a one-thread retained region it
 * is a valid no-op, while a retained inactive `if(false)` region is also a
 * team boundary. Standalone empty calls create neither a team nor a barrier;
 * nonempty work retains its ordinary workshare.
 */
[[nodiscard]] inline bool beginWorkshare(bool useful)
{
    if (!useful && inLexicalWorkshare())
    {
#pragma omp barrier
    }
    return useful;
}

/**
 * @brief Execute a worksharing lambda, creating a parallel region only if needed.
 *
 * If already inside any enclosing parallel region (see inLexicalWorkshare),
 * the lambda is called directly on the exact retained team. Otherwise, a new
 * parallel region is created.
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
        if (::llaminar::v2::inLexicalWorkshare())                  \
        {                                          \
            work_fn();                             \
        }                                          \
        else                                       \
        {                                          \
            ::llaminar::v2::noteOpenMPTeamFork();  \
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
        if (::llaminar::v2::inLexicalWorkshare())                  \
        {                                          \
            work_fn();                             \
            _Pragma("omp barrier")                 \
        }                                          \
        else                                       \
        {                                          \
            ::llaminar::v2::noteOpenMPTeamFork();  \
            _Pragma("omp parallel") { work_fn(); } \
        }                                          \
    } while (0)

/**
 * @brief Execute a worksharing lambda conditionally, with nested-safety.
 *
 * When we're already inside any enclosing parallel region (active, serialized
 * or inactive; inLexicalWorkshare), the lambda runs directly on the exact
 * retained team regardless of `condition`. Otherwise a new parallel region is
 * created when `condition` is true, or a 1-thread region when `condition` is
 * false.
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
        if (::llaminar::v2::inLexicalWorkshare())                                              \
        {                                                                      \
            /* Reuse the exact retained team, even serialized/inactive. */     \
            work_fn();                                                         \
        }                                                                      \
        else if (condition)                                                    \
        {                                                                      \
            ::llaminar::v2::noteOpenMPTeamFork();                              \
            _Pragma("omp parallel") { work_fn(); }                             \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            /* Serial: use 1-thread parallel so #pragma omp for still works */ \
            ::llaminar::v2::noteOpenMPTeamFork();                              \
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
     * @brief Check if currently executing within an active parallel region.
     *
     * Thin wrapper around omp_in_parallel() with the exact OpenMP meaning:
     * true when an ACTIVE enclosing region exists (an active ancestor can make
     * a nested-serialized one-thread team report true). Use
     * inLexicalWorkshare()/teamPlacement() for retained-team reuse decisions.
     *
     * @return true if enclosed by an active parallel region, false otherwise.
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
        if (::llaminar::v2::inLexicalWorkshare())                \
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
        if (::llaminar::v2::inLexicalWorkshare())                       \
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
 * kernel calls within a transformer layer. Inner kernels detect the retained
 * outer team lexically and reuse its exact threads instead of creating new
 * regions.
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
        if (::llaminar::v2::inLexicalWorkshare())                                        \
        {                                                                \
            /* Already in a retained region (nested call), just execute */ \
            body                                                         \
        }                                                                \
        else                                                             \
        {                                                                \
            ::llaminar::v2::noteOpenMPTeamFork();                        \
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
