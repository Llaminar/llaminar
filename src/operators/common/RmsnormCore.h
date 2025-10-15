/**
 * @file RmsnormCore.h
 * @brief Core (backend-agnostic) RMSNorm primitives used by both simple and MPI kernels.
 *
 * Centralizes the mathematical logic so higher-level kernels only provide
 * distribution, sharding and instrumentation. This avoids divergence in
 * epsilon handling, gamma indexing and future vectorization work.
 *
 * Threading policy:
 *  - Optional OpenMP parallelism over rows (if rows > 1 and problem size large).
 *  - Optional SIMD on the inner hidden-dimension reduction / application loops.
 *  - Guards against accidental nested parallel regions via omp_in_parallel().
 *
 * Extension points:
 *  - Fused residual (future): add variant taking residual pointer to accumulate.
 *  - Mixed precision: add template / overloads (fp16/bf16 inputs with float accumulate).
 *
 * @author David Sanftenberg
 */
#pragma once

#include <cstddef>
#include <vector>

namespace llaminar::kernels
{

    struct RMSNormExecOptions
    {
        bool allow_parallel = true;                  // Permit OpenMP parallel loops when heuristics match
        bool force_scalar = false;                   // Force fully scalar (single-thread, no SIMD pragma hints)
        std::size_t parallel_threshold_elems = 8192; // rows*cols threshold for row parallelism
        bool t5_compat_mode = false;                 // Use float32 accumulation for PyTorch T5LayerNorm parity
    };

    enum class GammaMode
    {
        REPLICATED,
        SHARDED
    };

    /**
     * @brief Reusable scratch buffers for RMSNorm fused path to avoid per-call allocations.
     *
     * Typical decode workloads invoke RMSNorm very frequently with rows=1; repeated
     * allocation of two small vectors shows up in profiles. This struct lets callers
     * (or an internal thread_local instance) reuse capacity. Capacity only grows.
     */
    struct RMSNormScratch
    {
        std::vector<double> row_sumsq; // size >= rows
        std::vector<float> inv;        // size >= rows

        void ensure(std::size_t rows)
        {
            if (row_sumsq.size() < rows)
                row_sumsq.resize(rows);
            if (inv.size() < rows)
                inv.resize(rows);
        }
        void clear()
        {
            // Does not free capacity; keeps memory hot for next invocation.
            row_sumsq.clear();
            inv.clear();
        }
    };

    // Compute per-row sum of squares (double precision accumulation).
    void rmsnorm_compute_row_sumsq(const float *src,
                                   std::size_t rows,
                                   std::size_t cols,
                                   double *row_sumsq,
                                   const RMSNormExecOptions &opts = {});

    // Convert row sum of squares into inverse RMS scaling factors.
    void rmsnorm_compute_inv(const double *row_sumsq,
                             std::size_t rows,
                             std::size_t cols,
                             float epsilon,
                             float *inv_out);

    // Apply (x * inv[row]) * gamma (mode dependent) writing to dst.
    void rmsnorm_apply(const float *src,
                       const float *gamma,
                       const float *inv,
                       std::size_t rows,
                       std::size_t cols,
                       float *dst,
                       GammaMode mode = GammaMode::REPLICATED,
                       std::size_t gamma_offset = 0,
                       const RMSNormExecOptions &opts = {});

    // Convenience fused helper: compute row_sumsq -> inv -> apply.
    void rmsnorm_row_major_fused(const float *src,
                                 const float *gamma,
                                 float *dst,
                                 std::size_t rows,
                                 std::size_t cols,
                                 float epsilon,
                                 GammaMode mode = GammaMode::REPLICATED,
                                 std::size_t gamma_offset = 0,
                                 const RMSNormExecOptions &opts = {});

    // Overload using caller-supplied scratch (required: scratch.ensure(rows) is handled internally)
    void rmsnorm_row_major_fused(const float *src,
                                 const float *gamma,
                                 float *dst,
                                 std::size_t rows,
                                 std::size_t cols,
                                 float epsilon,
                                 RMSNormScratch &scratch,
                                 GammaMode mode = GammaMode::REPLICATED,
                                 std::size_t gamma_offset = 0,
                                 const RMSNormExecOptions &opts = {});

} // namespace llaminar::kernels
