/**
 * @file CPURMSNormKernelT.cpp
 * @brief Typed RMSNorm kernel implementation with fused precision conversion
 *
 * Key optimization: Process data in row-sized chunks that fit in L1/L2 cache.
 * Never allocate full-sized FP32 scratch buffers - stream through the data.
 *
 * @author David Sanftenberg
 * @date 2025-12-04
 */

#include "CPURMSNormKernelT.h"
#include "../primitives/RMSNormPrimitives.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../utils/Logger.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <array>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../../utils/OpenMPUtils.h"

namespace llaminar2
{

    // =========================================================================
    // Constants for cache-friendly streaming and parallelization
    // =========================================================================

    // Maximum row size we support with stack-allocated scratch (16KB per row)
    // This covers d_model up to 4096 (16KB) which fits comfortably in L1 (32KB)
    // For larger models, we fall back to heap allocation per-thread
    static constexpr size_t MAX_STACK_ROW_SIZE = 4096;

    // Minimum number of rows to justify OpenMP parallelization overhead.
    // For single-row decode, thread spawning cost (~200µs) dominates the
    // actual computation (~2-5µs per row). Empirically determined threshold.
    static constexpr int MIN_ROWS_FOR_PARALLEL = 8;

    // Minimum total elements to justify parallelization for single-row case.
    // Even with 1 row, if d_model is huge (>8K), parallelization may help.
    static constexpr size_t MIN_ELEMENTS_FOR_PARALLEL = 65536; // 64K elements

    /**
     * @brief Determine if OpenMP parallelization is beneficial
     *
     * OpenMP thread spawning has fixed overhead (~100-200µs). For small workloads,
     * this overhead dominates the actual computation. We only parallelize when:
     * - Multiple rows (prefill) where work can be distributed, OR
     * - Single row with very large d_model where SIMD parallelism helps
     *
     * Note: This now returns true even when already in a parallel region,
     * because OMP_WORKSHARE_REGION will handle nested worksharing properly.
     *
     * @param rows Number of rows (sequence length)
     * @param cols Number of columns (d_model)
     * @return true if parallelization is likely beneficial
     */
    inline bool want_parallel(int rows, size_t cols)
    {
#ifdef _OPENMP
        // Multiple rows: parallelize if we have enough work per thread
        if (rows >= MIN_ROWS_FOR_PARALLEL)
            return true;

        // Single row: only parallelize for very large d_model
        size_t total_elements = static_cast<size_t>(rows) * cols;
        return total_elements >= MIN_ELEMENTS_FOR_PARALLEL;
#else
        (void)rows;
        (void)cols;
        return false;
#endif
    }

    // =========================================================================
    // FP32 Specialization Implementation
    // =========================================================================

    bool CPURMSNormKernelT<ActivationPrecision::FP32>::apply_typed(
        const float *input,
        const float *gamma,
        float *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        // Direct FP32 path - no conversion needed
        primitives::rmsnorm_fused_vectorized(
            input, gamma, output,
            static_cast<size_t>(rows),
            static_cast<size_t>(cols),
            epsilon);

        return true;
    }

    bool CPURMSNormKernelT<ActivationPrecision::FP32>::apply_with_residual_add(
        const float *residual,
        const float *fp32_input,
        const float *gamma,
        float *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!residual || !fp32_input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        const size_t ucols = static_cast<size_t>(cols);
        const bool use_parallel = want_parallel(rows, ucols);

        // Lambda for processing a single row
        auto process_row = [&](int row, float *scratch)
        {
            const float *res_row = residual + row * ucols;
            const float *in_row = fp32_input + row * ucols;
            float *out_row = output + row * ucols;

            // Fused add into scratch (stays in L1)
            for (size_t i = 0; i < ucols; ++i)
            {
                scratch[i] = res_row[i] + in_row[i];
            }

            // RMSNorm on single row (scratch → out_row)
            primitives::rmsnorm_fused_row_avx512(scratch, gamma, out_row, ucols, epsilon);
        };

        if (use_parallel)
        {
            // Parallel/worksharing path with thread-local scratch
            auto do_work = [&]()
            {
                // Thread-local scratch allocation
                std::array<float, MAX_STACK_ROW_SIZE> stack_scratch;
                std::vector<float> heap_scratch;
                float *scratch = (ucols <= MAX_STACK_ROW_SIZE)
                                     ? stack_scratch.data()
                                     : (heap_scratch.resize(ucols), heap_scratch.data());

#pragma omp for schedule(static)
                for (int row = 0; row < rows; ++row)
                {
                    process_row(row, scratch);
                }
            };
            OMP_WORKSHARE_REGION(do_work);
        }
        else
        {
            // Serial path: avoid thread spawning overhead for small workloads
            std::array<float, MAX_STACK_ROW_SIZE> stack_scratch;
            std::vector<float> heap_scratch;
            float *scratch = (ucols <= MAX_STACK_ROW_SIZE)
                                 ? stack_scratch.data()
                                 : (heap_scratch.resize(ucols), heap_scratch.data());

            for (int row = 0; row < rows; ++row)
            {
                process_row(row, scratch);
            }
        }

        return true;
    }

    // =========================================================================
    // BF16 Specialization Implementation (Cache-Streaming)
    // =========================================================================

    bool CPURMSNormKernelT<ActivationPrecision::BF16>::apply_typed(
        const uint16_t *input,
        const float *gamma,
        uint16_t *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        const size_t ucols = static_cast<size_t>(cols);
        const bool use_parallel = want_parallel(rows, ucols);

        // Lambda for processing a single row
        auto process_row = [&](int row, float *fp32_in, float *fp32_out)
        {
            const uint16_t *in_row = input + row * ucols;
            uint16_t *out_row = output + row * ucols;

            // 1. Dequantize BF16 → FP32 (one row, stays in L1)
            simd::convert_bf16_to_fp32(in_row, fp32_in, ucols);

            // 2. RMSNorm in FP32 (one row)
            primitives::rmsnorm_fused_row_avx512(fp32_in, gamma, fp32_out, ucols, epsilon);

            // 3. Quantize FP32 → BF16 (one row)
            simd::convert_fp32_to_bf16(fp32_out, out_row, ucols);
        };

        if (use_parallel)
        {
            auto do_work = [&]()
            {
                std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_in;
                std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
                std::vector<float> heap_fp32_in, heap_fp32_out;
                float *fp32_in = (ucols <= MAX_STACK_ROW_SIZE)
                                     ? stack_fp32_in.data()
                                     : (heap_fp32_in.resize(ucols), heap_fp32_in.data());
                float *fp32_out = (ucols <= MAX_STACK_ROW_SIZE)
                                      ? stack_fp32_out.data()
                                      : (heap_fp32_out.resize(ucols), heap_fp32_out.data());

#pragma omp for schedule(static)
                for (int row = 0; row < rows; ++row)
                {
                    process_row(row, fp32_in, fp32_out);
                }
            };
            OMP_WORKSHARE_REGION(do_work);
        }
        else
        {
            // Serial path for small workloads
            std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_in;
            std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
            std::vector<float> heap_fp32_in, heap_fp32_out;
            float *fp32_in = (ucols <= MAX_STACK_ROW_SIZE)
                                 ? stack_fp32_in.data()
                                 : (heap_fp32_in.resize(ucols), heap_fp32_in.data());
            float *fp32_out = (ucols <= MAX_STACK_ROW_SIZE)
                                  ? stack_fp32_out.data()
                                  : (heap_fp32_out.resize(ucols), heap_fp32_out.data());

            for (int row = 0; row < rows; ++row)
            {
                process_row(row, fp32_in, fp32_out);
            }
        }

        return true;
    }

    bool CPURMSNormKernelT<ActivationPrecision::BF16>::apply_with_residual_add(
        const uint16_t *residual,
        const float *fp32_input,
        const float *gamma,
        uint16_t *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!residual || !fp32_input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        const size_t ucols = static_cast<size_t>(cols);
        const bool use_parallel = want_parallel(rows, ucols);

        // Lambda for processing a single row
        auto process_row = [&](int row, float *fused, float *fp32_out)
        {
            const uint16_t *res_row = residual + row * ucols;
            const float *in_row = fp32_input + row * ucols;
            uint16_t *out_row = output + row * ucols;

            // 1. Fused dequant + add (one row): fused = dequant(res_bf16) + fp32_in
            simd::fused_bf16_residual_add(res_row, in_row, fused, ucols);

            // 2. RMSNorm in FP32 (one row)
            primitives::rmsnorm_fused_row_avx512(fused, gamma, fp32_out, ucols, epsilon);

            // 3. Quantize FP32 → BF16 (one row)
            simd::convert_fp32_to_bf16(fp32_out, out_row, ucols);
        };

        if (use_parallel)
        {
            auto do_work = [&]()
            {
                std::array<float, MAX_STACK_ROW_SIZE> stack_fused;
                std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
                std::vector<float> heap_fused, heap_fp32_out;
                float *fused = (ucols <= MAX_STACK_ROW_SIZE)
                                   ? stack_fused.data()
                                   : (heap_fused.resize(ucols), heap_fused.data());
                float *fp32_out = (ucols <= MAX_STACK_ROW_SIZE)
                                      ? stack_fp32_out.data()
                                      : (heap_fp32_out.resize(ucols), heap_fp32_out.data());

#pragma omp for schedule(static)
                for (int row = 0; row < rows; ++row)
                {
                    process_row(row, fused, fp32_out);
                }
            };
            OMP_WORKSHARE_REGION(do_work);
        }
        else
        {
            // Serial path for small workloads
            std::array<float, MAX_STACK_ROW_SIZE> stack_fused;
            std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
            std::vector<float> heap_fused, heap_fp32_out;
            float *fused = (ucols <= MAX_STACK_ROW_SIZE)
                               ? stack_fused.data()
                               : (heap_fused.resize(ucols), heap_fused.data());
            float *fp32_out = (ucols <= MAX_STACK_ROW_SIZE)
                                  ? stack_fp32_out.data()
                                  : (heap_fp32_out.resize(ucols), heap_fp32_out.data());

            for (int row = 0; row < rows; ++row)
            {
                process_row(row, fused, fp32_out);
            }
        }

        return true;
    }

    // =========================================================================
    // FP16 Specialization Implementation (Cache-Streaming)
    // =========================================================================

    bool CPURMSNormKernelT<ActivationPrecision::FP16>::apply_typed(
        const uint16_t *input,
        const float *gamma,
        uint16_t *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        const size_t ucols = static_cast<size_t>(cols);
        const bool use_parallel = want_parallel(rows, ucols);

        // Lambda for processing a single row
        auto process_row = [&](int row, float *fp32_in, float *fp32_out)
        {
            const uint16_t *in_row = input + row * ucols;
            uint16_t *out_row = output + row * ucols;

            // 1. Dequantize FP16 → FP32 (one row)
            simd::convert_fp16_to_fp32(in_row, fp32_in, ucols);

            // 2. RMSNorm in FP32 (one row)
            primitives::rmsnorm_fused_row_avx512(fp32_in, gamma, fp32_out, ucols, epsilon);

            // 3. Quantize FP32 → FP16 (one row)
            simd::convert_fp32_to_fp16(fp32_out, out_row, ucols);
        };

        if (use_parallel)
        {
            auto do_work = [&]()
            {
                std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_in;
                std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
                std::vector<float> heap_fp32_in, heap_fp32_out;
                float *fp32_in = (ucols <= MAX_STACK_ROW_SIZE)
                                     ? stack_fp32_in.data()
                                     : (heap_fp32_in.resize(ucols), heap_fp32_in.data());
                float *fp32_out = (ucols <= MAX_STACK_ROW_SIZE)
                                      ? stack_fp32_out.data()
                                      : (heap_fp32_out.resize(ucols), heap_fp32_out.data());

#pragma omp for schedule(static)
                for (int row = 0; row < rows; ++row)
                {
                    process_row(row, fp32_in, fp32_out);
                }
            };
            OMP_WORKSHARE_REGION(do_work);
        }
        else
        {
            // Serial path for small workloads
            std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_in;
            std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
            std::vector<float> heap_fp32_in, heap_fp32_out;
            float *fp32_in = (ucols <= MAX_STACK_ROW_SIZE)
                                 ? stack_fp32_in.data()
                                 : (heap_fp32_in.resize(ucols), heap_fp32_in.data());
            float *fp32_out = (ucols <= MAX_STACK_ROW_SIZE)
                                  ? stack_fp32_out.data()
                                  : (heap_fp32_out.resize(ucols), heap_fp32_out.data());

            for (int row = 0; row < rows; ++row)
            {
                process_row(row, fp32_in, fp32_out);
            }
        }

        return true;
    }

    bool CPURMSNormKernelT<ActivationPrecision::FP16>::apply_with_residual_add(
        const uint16_t *residual,
        const float *fp32_input,
        const float *gamma,
        uint16_t *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!residual || !fp32_input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        const size_t ucols = static_cast<size_t>(cols);
        const bool use_parallel = want_parallel(rows, ucols);

        // Lambda for processing a single row
        auto process_row = [&](int row, float *fused, float *fp32_out)
        {
            const uint16_t *res_row = residual + row * ucols;
            const float *in_row = fp32_input + row * ucols;
            uint16_t *out_row = output + row * ucols;

            // 1. Fused dequant + add (one row)
            simd::fused_fp16_residual_add(res_row, in_row, fused, ucols);

            // 2. RMSNorm in FP32 (one row)
            primitives::rmsnorm_fused_row_avx512(fused, gamma, fp32_out, ucols, epsilon);

            // 3. Quantize FP32 → FP16 (one row)
            simd::convert_fp32_to_fp16(fp32_out, out_row, ucols);
        };

        if (use_parallel)
        {
            auto do_work = [&]()
            {
                std::array<float, MAX_STACK_ROW_SIZE> stack_fused;
                std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
                std::vector<float> heap_fused, heap_fp32_out;
                float *fused = (ucols <= MAX_STACK_ROW_SIZE)
                                   ? stack_fused.data()
                                   : (heap_fused.resize(ucols), heap_fused.data());
                float *fp32_out = (ucols <= MAX_STACK_ROW_SIZE)
                                      ? stack_fp32_out.data()
                                      : (heap_fp32_out.resize(ucols), heap_fp32_out.data());

#pragma omp for schedule(static)
                for (int row = 0; row < rows; ++row)
                {
                    process_row(row, fused, fp32_out);
                }
            };
            OMP_WORKSHARE_REGION(do_work);
        }
        else
        {
            // Serial path for small workloads
            std::array<float, MAX_STACK_ROW_SIZE> stack_fused;
            std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
            std::vector<float> heap_fused, heap_fp32_out;
            float *fused = (ucols <= MAX_STACK_ROW_SIZE)
                               ? stack_fused.data()
                               : (heap_fused.resize(ucols), heap_fused.data());
            float *fp32_out = (ucols <= MAX_STACK_ROW_SIZE)
                                  ? stack_fp32_out.data()
                                  : (heap_fp32_out.resize(ucols), heap_fp32_out.data());

            for (int row = 0; row < rows; ++row)
            {
                process_row(row, fused, fp32_out);
            }
        }

        return true;
    }

    // =========================================================================
    // Q8_1 Specialization Implementation (Cache-Streaming)
    // =========================================================================

    bool CPURMSNormKernelT<ActivationPrecision::Q8_1>::apply_typed(
        const Q8_1Block *input,
        const float *gamma,
        Q8_1Block *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        // Q8_1 requires cols to be multiple of 32 (block size)
        if (cols % 32 != 0)
        {
            LOG_ERROR("Q8_1 RMSNorm requires cols to be multiple of 32, got " << cols);
            return false;
        }

        const size_t ucols = static_cast<size_t>(cols);
        const size_t blocks_per_row = ucols / 32;

        // Use the optimized pure-integer primitive with dynamic threading
        primitives::RMSNormExecOptions opts;
        opts.allow_parallel = want_parallel(rows, ucols);

        primitives::rmsnorm_q8_1_pure_integer(
            input, gamma, output,
            static_cast<size_t>(rows), blocks_per_row,
            epsilon, opts);

        return true;
    }

    bool CPURMSNormKernelT<ActivationPrecision::Q8_1>::apply_with_residual_add(
        const Q8_1Block *residual,
        const float *fp32_input,
        const float *gamma,
        Q8_1Block *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!residual || !fp32_input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        // Q8_1 requires cols to be multiple of 32 (block size)
        if (cols % 32 != 0)
        {
            LOG_ERROR("Q8_1 RMSNorm requires cols to be multiple of 32, got " << cols);
            return false;
        }

        const size_t ucols = static_cast<size_t>(cols);
        const size_t blocks_per_row = ucols / 32;
        const bool use_parallel = want_parallel(rows, ucols);

        // Lambda for processing a single row
        auto process_row = [&](int row, float *fused, float *fp32_out)
        {
            const Q8_1Block *res_row = residual + row * blocks_per_row;
            const float *in_row = fp32_input + row * ucols;
            Q8_1Block *out_row = output + row * blocks_per_row;

            // 1. Fused dequant + add (one row)
            simd::fused_q8_1_residual_add(res_row, in_row, fused, ucols);

            // 2. RMSNorm in FP32 (one row)
            primitives::rmsnorm_fused_row_avx512(fused, gamma, fp32_out, ucols, epsilon);

            // 3. Quantize FP32 → Q8_1 (one row)
            simd::quantize_fp32_to_q8_1_blocks(fp32_out, out_row, ucols);
        };

        if (use_parallel)
        {
            auto do_work = [&]()
            {
                std::array<float, MAX_STACK_ROW_SIZE> stack_fused;
                std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
                std::vector<float> heap_fused, heap_fp32_out;
                float *fused = (ucols <= MAX_STACK_ROW_SIZE)
                                   ? stack_fused.data()
                                   : (heap_fused.resize(ucols), heap_fused.data());
                float *fp32_out = (ucols <= MAX_STACK_ROW_SIZE)
                                      ? stack_fp32_out.data()
                                      : (heap_fp32_out.resize(ucols), heap_fp32_out.data());

#pragma omp for schedule(static)
                for (int row = 0; row < rows; ++row)
                {
                    process_row(row, fused, fp32_out);
                }
            };
            OMP_WORKSHARE_REGION(do_work);
        }
        else
        {
            // Serial path for small workloads
            std::array<float, MAX_STACK_ROW_SIZE> stack_fused;
            std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
            std::vector<float> heap_fused, heap_fp32_out;
            float *fused = (ucols <= MAX_STACK_ROW_SIZE)
                               ? stack_fused.data()
                               : (heap_fused.resize(ucols), heap_fused.data());
            float *fp32_out = (ucols <= MAX_STACK_ROW_SIZE)
                                  ? stack_fp32_out.data()
                                  : (heap_fp32_out.resize(ucols), heap_fp32_out.data());

            for (int row = 0; row < rows; ++row)
            {
                process_row(row, fused, fp32_out);
            }
        }

        return true;
    }

    // =========================================================================
    // Q8_1 FP32 Path Implementation (for mutable Q8_1 tensors)
    // =========================================================================

    /**
     * @brief FP32 path for Q8_1 kernel - supports mutable Q8_1 tensors
     *
     * When Q8_1 tensors are used as mutable activation buffers, they store
     * FP32 data in a dequant cache rather than actual Q8_1 blocks. This
     * method enables RMSNorm to work with such tensors by using the same
     * vectorized FP32 primitive as the FP32 specialization.
     */
    bool CPURMSNormKernelT<ActivationPrecision::Q8_1>::apply(
        const float *input,
        const float *gamma,
        float *output,
        int rows,
        int cols,
        float epsilon,
        bool use_bf16,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)use_bf16;
        (void)mpi_ctx;
        (void)device_idx;

        if (!input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        // Use the same FP32 primitive as the FP32 specialization
        primitives::rmsnorm_fused_vectorized(
            input, gamma, output,
            static_cast<size_t>(rows),
            static_cast<size_t>(cols),
            epsilon);

        return true;
    }

    // =========================================================================
    // Q16_1 Specialization (Q16_1 input → FP32 output)
    // =========================================================================

    bool CPURMSNormKernelT<ActivationPrecision::Q16_1>::apply_typed(
        const Q16_1Block *input,
        const float *gamma,
        float *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        // Q16_1 requires cols to be multiple of 32 (block size)
        if (cols % 32 != 0)
        {
            LOG_ERROR("Q16_1 RMSNorm requires cols to be multiple of 32, got " << cols);
            return false;
        }

        const size_t ucols = static_cast<size_t>(cols);

        // Use the Q16_1→FP32 primitive with automatic parallelization
        primitives::RMSNormExecOptions opts;
        opts.allow_parallel = want_parallel(rows, ucols);

        primitives::rmsnorm_q16_1_fp32_fused(
            input, gamma, output,
            static_cast<size_t>(rows), ucols,
            epsilon, opts);

        return true;
    }

} // namespace llaminar2
