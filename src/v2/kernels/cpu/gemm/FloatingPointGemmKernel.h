/**
 * @file FloatingPointGemmKernel.h
 * @brief ITensorGemm implementation for floating-point GEMM using OneDNN
 *
 * Provides optimized GEMM for homogeneous floating-point type combinations:
 * - FP32 weights × FP32 activations → FP32 output
 * - FP16 weights × FP16 activations → FP32 output
 * - BF16 weights × BF16 activations → FP32 output
 *
 * For quantized weight GEMM (Q4_0, Q8_0, Q8_1, etc.), use CPUQuantisedGemmKernel.
 *
 * @author David Sanftenberg
 * @date 2025-11-26
 */

#pragma once

#include "memory/CPUWeightStoragePlacement.h"

#ifndef HAVE_ONEDNN
#error "OneDNN support is required for FloatingPointGemmKernel"
#endif

#include <oneapi/dnnl/dnnl.hpp>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_map>

#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/FP16Utils.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../common/FloatingExpertNumericalContract.h"
#include "../CPUKernelBase.h"
#include "../primitives/SwiGLUPrimitives.h"
#include "../primitives/SoftmaxPrimitives_New.h"

namespace llaminar2
{
    namespace gemm
    {
        // ========== OneDNN Engine/Stream Singletons ==========

        /**
         * @brief Get thread-local OneDNN CPU engine
         */
        inline dnnl::engine &onednn_engine()
        {
            static thread_local dnnl::engine engine_instance(dnnl::engine::kind::cpu, 0);
            return engine_instance;
        }

        /**
         * @brief Get thread-local OneDNN execution stream
         */
        inline dnnl::stream &onednn_stream()
        {
            static thread_local dnnl::stream stream_instance(onednn_engine());
            return stream_instance;
        }

        // ========== OneDNN GEMM Primitives ==========

        inline bool run_fp32_skinny_matmul(const float *A,
                                           const float *B,
                                           float *C,
                                           int M,
                                           int N,
                                           int K,
                                           bool transpose_B,
                                           float alpha = 1.0f,
                                           float beta = 0.0f,
                                           const float *bias = nullptr)
        {
            if (!A || !B || !C || M < 0 || N < 0 || K < 0)
            {
                LOG_ERROR("[FloatingPointGemmKernel] Invalid skinny FP32 matmul pointers/dims: "
                          << "A=" << static_cast<const void *>(A)
                          << " B=" << static_cast<const void *>(B)
                          << " C=" << static_cast<void *>(C)
                          << " M=" << M << " N=" << N << " K=" << K);
                return false;
            }

            /*
             * MTP verifier rows need decode-equivalent FP32 accumulation, so we
             * keep the scalar kk order used by one-row decode. The ordinary
             * row/column loop below is correct, but it reloads the same B row
             * once per verifier row. This grouped-column path tiles runtime M
             * in fixed four-row groups and reuses that B row while maintaining
             * one independent scalar accumulator per row. The fixed tile bounds
             * register pressure for deep speculation; it is intentionally not a
             * vector reduction because changing K reduction order would risk
             * recurrent-state drift in GDN/short-conv publication.
             */
            if (transpose_B && M >= 2 && N >= 128)
            {
                constexpr int kVerifierRowTile = 4;
                const int row_tiles =
                    (M + kVerifierRowTile - 1) / kVerifierRowTile;
                auto grouped_column_work = [&]()
                {
#pragma omp for collapse(2) schedule(static)
                    for (int row_tile = 0; row_tile < row_tiles; ++row_tile)
                    {
                        for (int col = 0; col < N; ++col)
                        {
                            const int first_row = row_tile * kVerifierRowTile;
                            const int tile_rows =
                                std::min(kVerifierRowTile, M - first_row);
                            const float *b_row =
                                B + static_cast<size_t>(col) * K;
                            float accumulators[kVerifierRowTile] = {};
                            const float *a0 =
                                A + static_cast<size_t>(first_row) * K;

                            /*
                             * Keep explicit accumulators for each physical tile
                             * width. A runtime inner row loop invites the compiler
                             * to choose a different vector/FMA schedule once M
                             * exceeds four, which is mathematically valid but not
                             * byte-identical to the proven M=1 decode expression.
                             */
                            if (tile_rows == 1)
                            {
                                float acc0 = 0.0f;
                                for (int kk = 0; kk < K; ++kk)
                                    acc0 += a0[kk] * b_row[kk];
                                accumulators[0] = acc0;
                            }
                            else if (tile_rows == 2)
                            {
                                const float *a1 = a0 + K;
                                float acc0 = 0.0f;
                                float acc1 = 0.0f;
                                for (int kk = 0; kk < K; ++kk)
                                {
                                    const float b = b_row[kk];
                                    acc0 += a0[kk] * b;
                                    acc1 += a1[kk] * b;
                                }
                                accumulators[0] = acc0;
                                accumulators[1] = acc1;
                            }
                            else if (tile_rows == 3)
                            {
                                const float *a1 = a0 + K;
                                const float *a2 = a1 + K;
                                float acc0 = 0.0f;
                                float acc1 = 0.0f;
                                float acc2 = 0.0f;
                                for (int kk = 0; kk < K; ++kk)
                                {
                                    const float b = b_row[kk];
                                    acc0 += a0[kk] * b;
                                    acc1 += a1[kk] * b;
                                    acc2 += a2[kk] * b;
                                }
                                accumulators[0] = acc0;
                                accumulators[1] = acc1;
                                accumulators[2] = acc2;
                            }
                            else
                            {
                                const float *a1 = a0 + K;
                                const float *a2 = a1 + K;
                                const float *a3 = a2 + K;
                                float acc0 = 0.0f;
                                float acc1 = 0.0f;
                                float acc2 = 0.0f;
                                float acc3 = 0.0f;
                                for (int kk = 0; kk < K; ++kk)
                                {
                                    const float b = b_row[kk];
                                    acc0 += a0[kk] * b;
                                    acc1 += a1[kk] * b;
                                    acc2 += a2[kk] * b;
                                    acc3 += a3[kk] * b;
                                }
                                accumulators[0] = acc0;
                                accumulators[1] = acc1;
                                accumulators[2] = acc2;
                                accumulators[3] = acc3;
                            }

                            const float bias_value = bias ? bias[col] : 0.0f;
                            for (int tile_row = 0; tile_row < tile_rows; ++tile_row)
                            {
                                const size_t output_index =
                                    static_cast<size_t>(first_row + tile_row) * N + col;
                                float value =
                                    alpha * accumulators[tile_row] + bias_value;
                                if (beta != 0.0f)
                                    value += beta * C[output_index];
                                C[output_index] = value;
                            }
                        }
                    }
                };
                OMP_WORKSHARE_REGION(grouped_column_work);
                return true;
            }

            auto work = [&]()
            {
#pragma omp for collapse(2) schedule(static)
                for (int row = 0; row < M; ++row)
                {
                    for (int col = 0; col < N; ++col)
                    {
                        float acc = 0.0f;
                        const float *a_row = A + static_cast<size_t>(row) * K;
                        if (transpose_B)
                        {
                            const float *b_row = B + static_cast<size_t>(col) * K;
                            for (int kk = 0; kk < K; ++kk)
                                acc += a_row[kk] * b_row[kk];
                        }
                        else
                        {
                            for (int kk = 0; kk < K; ++kk)
                                acc += a_row[kk] * B[static_cast<size_t>(kk) * N + col];
                        }

                        float value = alpha * acc;
                        if (bias)
                            value += bias[col];
                        if (beta != 0.0f)
                            value += beta * C[static_cast<size_t>(row) * N + col];
                        C[static_cast<size_t>(row) * N + col] = value;
                    }
                }
            };
            OMP_WORKSHARE_REGION(work);
            return true;
        }

        /**
         * @brief Execute a decode-equivalent small-M GEMM for 16-bit float storage.
         *
         * MTP verifier publication compares grouped candidate rows against the
         * exact M=1 decode path.  Library BF16/FP16 GEMMs are allowed to choose
         * different vector widths or reduction trees when M changes, so verifier
         * rows use this deliberately simple primitive.  It converts each 16-bit
         * activation/weight element to FP32 at the same point in the K loop and
         * accumulates in scalar FP32, matching the serial decode dot-product
         * order while still grouping rows and columns inside one OpenMP region.
         */
        template <typename DecodeFn>
        inline bool run_16bit_skinny_matmul(const uint16_t *A,
                                            const uint16_t *B,
                                            float *C,
                                            int M,
                                            int N,
                                            int K,
                                            bool transpose_B,
                                            DecodeFn decode,
                                            float alpha = 1.0f,
                                            float beta = 0.0f)
        {
            if (!A || !B || !C || M < 0 || N < 0 || K < 0)
            {
                LOG_ERROR("[FloatingPointGemmKernel] Invalid skinny 16-bit matmul pointers/dims: "
                          << "A=" << static_cast<const void *>(A)
                          << " B=" << static_cast<const void *>(B)
                          << " C=" << static_cast<void *>(C)
                          << " M=" << M << " N=" << N << " K=" << K);
                return false;
            }

            /*
             * Four-row tiles reuse each decoded weight value without joining
             * the per-row accumulators. Every accumulator still visits K in
             * increasing order, exactly as the one-row decode primitive does.
             */
            constexpr int kVerifierRowTile = 4;
            const int row_tiles =
                (M + kVerifierRowTile - 1) / kVerifierRowTile;
            auto work = [&]()
            {
#pragma omp for collapse(2) schedule(static)
                for (int row_tile = 0; row_tile < row_tiles; ++row_tile)
                {
                    for (int col = 0; col < N; ++col)
                    {
                        const int first_row = row_tile * kVerifierRowTile;
                        const int tile_rows =
                            std::min(kVerifierRowTile, M - first_row);
                        float accumulators[kVerifierRowTile] = {};
                        for (int kk = 0; kk < K; ++kk)
                        {
                            const uint16_t b_bits = transpose_B
                                                        ? B[static_cast<size_t>(col) * K + kk]
                                                        : B[static_cast<size_t>(kk) * N + col];
                            const float b = decode(b_bits);
                            for (int tile_row = 0; tile_row < tile_rows; ++tile_row)
                            {
                                const uint16_t *a_row =
                                    A + static_cast<size_t>(first_row + tile_row) * K;
                                accumulators[tile_row] += decode(a_row[kk]) * b;
                            }
                        }

                        for (int tile_row = 0; tile_row < tile_rows; ++tile_row)
                        {
                            const size_t output_index =
                                static_cast<size_t>(first_row + tile_row) * N + col;
                            float value = alpha * accumulators[tile_row];
                            if (beta != 0.0f)
                                value += beta * C[output_index];
                            C[output_index] = value;
                        }
                    }
                }
            };
            OMP_WORKSHARE_REGION(work);
            return true;
        }

        inline bool run_fp16_skinny_matmul(const uint16_t *A,
                                           const uint16_t *B,
                                           float *C,
                                           int M,
                                           int N,
                                           int K,
                                           bool transpose_B,
                                           float alpha = 1.0f,
                                           float beta = 0.0f)
        {
            return run_16bit_skinny_matmul(
                A, B, C, M, N, K, transpose_B,
                [](uint16_t value)
                {
                    return fp16_to_fp32(value);
                },
                alpha,
                beta);
        }

        inline bool run_bf16_skinny_matmul(const uint16_t *A,
                                           const uint16_t *B,
                                           float *C,
                                           int M,
                                           int N,
                                           int K,
                                           bool transpose_B,
                                           float alpha = 1.0f,
                                           float beta = 0.0f)
        {
            return run_16bit_skinny_matmul(
                A, B, C, M, N, K, transpose_B,
                [](uint16_t value)
                {
                    return simd::bf16_to_fp32(value);
                },
                alpha,
                beta);
        }

        /**
         * @brief Decode-equivalent small-M GEMM for FP32 activations and 16-bit weights.
         *
         * Floating FP16/BF16 model weights still commonly feed FP32 hidden rows
         * in verifier publication code.  This helper is the CPU counterpart to
         * the GPU fp32x16 verifier kernels: it keeps A in FP32, converts each
         * 16-bit weight element inside the scalar K loop, and therefore gives
         * M=1 and grouped runtime-M rows one shared reduction contract.
         */
        template <typename DecodeFn>
        inline bool run_fp32x16_skinny_matmul(const float *A,
                                              const uint16_t *B,
                                              float *C,
                                              int M,
                                              int N,
                                              int K,
                                              bool transpose_B,
                                              DecodeFn decode,
                                              float alpha = 1.0f,
                                              float beta = 0.0f)
        {
            if (!A || !B || !C || M < 0 || N < 0 || K < 0)
            {
                LOG_ERROR("[FloatingPointGemmKernel] Invalid skinny FP32x16 matmul pointers/dims: "
                          << "A=" << static_cast<const void *>(A)
                          << " B=" << static_cast<const void *>(B)
                          << " C=" << static_cast<void *>(C)
                          << " M=" << M << " N=" << N << " K=" << K);
                return false;
            }

            /*
             * Match the homogeneous 16-bit primitive above: a fixed four-row
             * tile shares each decoded weight while preserving one independent
             * scalar K reduction per verifier row.
             */
            constexpr int kVerifierRowTile = 4;
            const int row_tiles =
                (M + kVerifierRowTile - 1) / kVerifierRowTile;
            auto work = [&]()
            {
#pragma omp for collapse(2) schedule(static)
                for (int row_tile = 0; row_tile < row_tiles; ++row_tile)
                {
                    for (int col = 0; col < N; ++col)
                    {
                        const int first_row = row_tile * kVerifierRowTile;
                        const int tile_rows =
                            std::min(kVerifierRowTile, M - first_row);
                        float accumulators[kVerifierRowTile] = {};
                        for (int kk = 0; kk < K; ++kk)
                        {
                            const uint16_t b_bits = transpose_B
                                                        ? B[static_cast<size_t>(col) * K + kk]
                                                        : B[static_cast<size_t>(kk) * N + col];
                            const float b = decode(b_bits);
                            for (int tile_row = 0; tile_row < tile_rows; ++tile_row)
                            {
                                const float *a_row =
                                    A + static_cast<size_t>(first_row + tile_row) * K;
                                accumulators[tile_row] += a_row[kk] * b;
                            }
                        }

                        for (int tile_row = 0; tile_row < tile_rows; ++tile_row)
                        {
                            const size_t output_index =
                                static_cast<size_t>(first_row + tile_row) * N + col;
                            float value = alpha * accumulators[tile_row];
                            if (beta != 0.0f)
                                value += beta * C[output_index];
                            C[output_index] = value;
                        }
                    }
                }
            };
            OMP_WORKSHARE_REGION(work);
            return true;
        }

        inline bool run_fp32xfp16_skinny_matmul(const float *A,
                                                const uint16_t *B,
                                                float *C,
                                                int M,
                                                int N,
                                                int K,
                                                bool transpose_B,
                                                float alpha = 1.0f,
                                                float beta = 0.0f)
        {
            return run_fp32x16_skinny_matmul(
                A, B, C, M, N, K, transpose_B,
                [](uint16_t value)
                {
                    return fp16_to_fp32(value);
                },
                alpha,
                beta);
        }

        inline bool run_fp32xbf16_skinny_matmul(const float *A,
                                                const uint16_t *B,
                                                float *C,
                                                int M,
                                                int N,
                                                int K,
                                                bool transpose_B,
                                                float alpha = 1.0f,
                                                float beta = 0.0f)
        {
            return run_fp32x16_skinny_matmul(
                A, B, C, M, N, K, transpose_B,
                [](uint16_t value)
                {
                    return simd::bf16_to_fp32(value);
                },
                alpha,
                beta);
        }

        /**
         * @brief Decode one GPU-aligned floating-expert weight without changing bits.
         *
         * @tparam WeightType Native floating tensor storage type.
         * @param weights Contiguous row-major weight storage.
         * @param index Element index in that storage.
         * @return The exactly represented FP32 value used by device kernels.
         */
        template <TensorType WeightType>
        inline float load_gpu_aligned_expert_weight_scalar(
            const void *weights,
            std::size_t index) noexcept
        {
            static_assert(
                WeightType == TensorType::FP32 ||
                    WeightType == TensorType::FP16 ||
                    WeightType == TensorType::BF16,
                "GPU-aligned expert weights must use a floating storage type");
            if constexpr (WeightType == TensorType::FP32)
                return static_cast<const float *>(weights)[index];
            else if constexpr (WeightType == TensorType::FP16)
                return fp16_to_fp32(
                    static_cast<const std::uint16_t *>(weights)[index]);
            else
                return simd::bf16_to_fp32(
                    static_cast<const std::uint16_t *>(weights)[index]);
        }

#if defined(__AVX512F__) && defined(__AVX512BW__)
        /**
         * @brief Load sixteen contiguous GPU-aligned expert weights as exact FP32 lanes.
         *
         * BF16 conversion is the format's exact left shift. FP16 uses the ISA's
         * IEEE conversion instruction; both produce the same finite binary32
         * words as the CUDA/HIP load helpers. Masked tail lanes become positive
         * zero, matching the initialized inactive GPU reduction lanes.
         *
         * @tparam WeightType Native floating tensor storage type.
         * @param weights Contiguous row-major weight storage.
         * @param index First element index.
         * @param mask Active lanes for a partial K tail.
         * @return Sixteen FP32 weight lanes.
         */
        template <TensorType WeightType>
        inline __m512 load_gpu_aligned_expert_weight_avx512(
            const void *weights,
            std::size_t index,
            __mmask16 mask) noexcept
        {
            static_assert(
                WeightType == TensorType::FP32 ||
                    WeightType == TensorType::FP16 ||
                    WeightType == TensorType::BF16,
                "GPU-aligned expert weights must use a floating storage type");
            if constexpr (WeightType == TensorType::FP32)
            {
                return _mm512_maskz_loadu_ps(
                    mask,
                    static_cast<const float *>(weights) + index);
            }
            else
            {
                const __m256i packed = _mm256_maskz_loadu_epi16(
                    mask,
                    static_cast<const std::uint16_t *>(weights) + index);
                if constexpr (WeightType == TensorType::FP16)
                    return _mm512_cvtph_ps(packed);
                else
                    return _mm512_castsi512_ps(
                        _mm512_slli_epi32(
                            _mm512_cvtepu16_epi32(packed), 16));
            }
        }

        /**
         * @brief Reduce the 256 logical GPU lanes with the identical add tree.
         *
         * Wide instructions process independent nodes from one tree level; a
         * store/load boundary separates levels. The last four levels use the
         * scalar explicit-rounding primitive. Consequently vector width changes
         * throughput without changing a single parenthesization or FP32 edge.
         *
         * @param partials Exactly 256 lane accumulators.
         * @return Root word of the fixed reduction tree.
         */
        inline float reduce_gpu_aligned_expert_partials_avx512(
            float *partials) noexcept
        {
            constexpr int kLanes =
                floating_expert_numerical_contract::kReductionLanes;
            for (int stride = kLanes / 2; stride >= 16; stride >>= 1)
            {
                for (int lane = 0; lane < stride; lane += 16)
                {
                    const __m512 lhs = _mm512_load_ps(partials + lane);
                    const __m512 rhs =
                        _mm512_load_ps(partials + lane + stride);
                    _mm512_store_ps(
                        partials + lane,
                        _mm512_add_ps(lhs, rhs));
                }
            }
            for (int stride = 8; stride > 0; stride >>= 1)
            {
                for (int lane = 0; lane < stride; ++lane)
                {
                    partials[lane] = device_fp32_contract::add(
                        partials[lane], partials[lane + stride]);
                }
            }
            return partials[0];
        }

        /**
         * @brief Compute one output column for a fixed number of rows with
         *        enough independent AVX-512 chains to cover FMA latency.
         *
         * The device contract owns 256 independent logical accumulators.  A
         * naive CPU translation completes all K contributions for one vector
         * of sixteen lanes before starting the next vector, leaving only
         * `Rows` dependent FMA chains in flight.  This schedule advances up to
         * sixteen chains together.  Each individual lane still observes the
         * identical increasing `base` order, and the final reduction uses the
         * unchanged device tree, so the optimization cannot alter a result bit.
         *
         * @tparam WeightType Native floating tensor storage type.
         * @tparam Rows Compile-time row count in `[1,4]`.
         * @param activations First activation row for this tile.
         * @param weights Complete row-major transposed weight tensor.
         * @param weight_row Element offset of the selected output column.
         * @param k Reduction width and activation-row stride.
         * @param reduced Exact tree root for each row.
         */
        template <TensorType WeightType, int Rows>
        inline void dot_gpu_aligned_expert_rows_avx512(
            const float *activations,
            const void *weights,
            std::size_t weight_row,
            int k,
            float *reduced) noexcept
        {
            static_assert(Rows >= 1 && Rows <= 4);
            constexpr int kLanes =
                floating_expert_numerical_contract::kReductionLanes;
            constexpr int kVectorLanes = 16;
            // Bound live accumulators to sixteen, leaving ample registers for
            // the current weights, activation, masks, and address arithmetic.
            constexpr int kLaneBlocksPerTile = 16 / Rows;
            alignas(64) float partials[Rows][kLanes] = {};

            for (int lane_tile = 0; lane_tile < kLanes;
                 lane_tile += kVectorLanes * kLaneBlocksPerTile)
            {
                __m512 accumulators[Rows][kLaneBlocksPerTile];
                // These dimensions are compile-time constants.  Explicit
                // unrolling is required here: otherwise GCC leaves the array
                // compiler-indexed and reloads every accumulator from the
                // stack inside the FMA loop.
#pragma GCC unroll 4
                for (int row = 0; row < Rows; ++row)
                {
#pragma GCC unroll 16
                    for (int block = 0; block < kLaneBlocksPerTile; ++block)
                        accumulators[row][block] = _mm512_setzero_ps();
                }

                // Advancing base outside the lane-block loop exposes the
                // independent chains together and reads neighboring cache
                // lines before moving to the next 256-lane K stripe.
                for (int base = 0; base < k; base += kLanes)
                {
#pragma GCC unroll 16
                    for (int block = 0; block < kLaneBlocksPerTile; ++block)
                    {
                        const int lane_block =
                            lane_tile + block * kVectorLanes;
                        if (lane_block >= kLanes)
                            continue;
                        const int first_k = base + lane_block;
                        if (first_k >= k)
                            continue;
                        const int valid = std::min(kVectorLanes, k - first_k);
                        const __mmask16 mask =
                            valid == kVectorLanes
                                ? static_cast<__mmask16>(0xffffu)
                                : static_cast<__mmask16>(
                                      (std::uint32_t{1} << valid) - 1u);
                        const __m512 weight_values =
                            load_gpu_aligned_expert_weight_avx512<WeightType>(
                                weights,
                                weight_row +
                                    static_cast<std::size_t>(first_k),
                                mask);
#pragma GCC unroll 4
                        for (int row = 0; row < Rows; ++row)
                        {
                            const float *activation_values =
                                activations +
                                static_cast<std::size_t>(row) * k +
                                static_cast<std::size_t>(first_k);
                            accumulators[row][block] = _mm512_fmadd_ps(
                                _mm512_maskz_loadu_ps(mask, activation_values),
                                weight_values,
                                accumulators[row][block]);
                        }
                    }
                }

#pragma GCC unroll 4
                for (int row = 0; row < Rows; ++row)
                {
#pragma GCC unroll 16
                    for (int block = 0; block < kLaneBlocksPerTile; ++block)
                    {
                        const int lane_block =
                            lane_tile + block * kVectorLanes;
                        if (lane_block < kLanes)
                        {
                            _mm512_store_ps(
                                partials[row] + lane_block,
                                accumulators[row][block]);
                        }
                    }
                }
            }

            for (int row = 0; row < Rows; ++row)
            {
                reduced[row] =
                    reduce_gpu_aligned_expert_partials_avx512(partials[row]);
            }
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief Load eight contiguous GPU-aligned expert weights as exact FP32 lanes.
         *
         * AVX2 has no masked 16-bit load, so the uncommon partial-K tail is
         * copied into a zeroed stack word before conversion. Production Qwen
         * expert dimensions are multiples of 256 and remain on the direct-load
         * branch. The conversion instructions preserve the same finite FP32
         * words as the scalar and device loaders.
         *
         * @tparam WeightType Native floating tensor storage type.
         * @param weights Contiguous row-major weight storage.
         * @param index First element index.
         * @param valid Number of active lanes in `[1,8]`.
         * @return Eight FP32 weight lanes with inactive lanes set to zero.
         */
        template <TensorType WeightType>
        inline __m256 load_gpu_aligned_expert_weight_avx2(
            const void *weights,
            std::size_t index,
            int valid) noexcept
        {
            static_assert(
                WeightType == TensorType::FP32 ||
                    WeightType == TensorType::FP16 ||
                    WeightType == TensorType::BF16,
                "GPU-aligned expert weights must use a floating storage type");
            if constexpr (WeightType == TensorType::FP32)
            {
                if (valid == 8)
                {
                    return _mm256_loadu_ps(
                        static_cast<const float *>(weights) + index);
                }
                alignas(32) float tail[8] = {};
                std::memcpy(
                    tail,
                    static_cast<const float *>(weights) + index,
                    static_cast<std::size_t>(valid) * sizeof(float));
                return _mm256_load_ps(tail);
            }
            else
            {
                __m128i packed{};
                if (valid == 8)
                {
                    packed = _mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(
                            static_cast<const std::uint16_t *>(weights) +
                            index));
                }
                else
                {
                    alignas(16) std::uint16_t tail[8] = {};
                    std::memcpy(
                        tail,
                        static_cast<const std::uint16_t *>(weights) + index,
                        static_cast<std::size_t>(valid) *
                            sizeof(std::uint16_t));
                    packed = _mm_load_si128(
                        reinterpret_cast<const __m128i *>(tail));
                }
                if constexpr (WeightType == TensorType::FP16)
                    return _mm256_cvtph_ps(packed);
                else
                    return _mm256_castsi256_ps(
                        _mm256_slli_epi32(
                            _mm256_cvtepu16_epi32(packed), 16));
            }
        }

        /**
         * @brief Load eight FP32 activations without crossing a K-row tail.
         * @param activations Address of the first active activation.
         * @param valid Number of active lanes in `[1,8]`.
         * @return Eight activation lanes with inactive lanes set to zero.
         */
        inline __m256 load_gpu_aligned_expert_activation_avx2(
            const float *activations,
            int valid) noexcept
        {
            if (valid == 8)
                return _mm256_loadu_ps(activations);
            alignas(32) float tail[8] = {};
            std::memcpy(
                tail,
                activations,
                static_cast<std::size_t>(valid) * sizeof(float));
            return _mm256_load_ps(tail);
        }

        /**
         * @brief Reduce the 256 logical GPU lanes using AVX2 tree nodes.
         * @param partials Exactly 256 lane accumulators.
         * @return Root word of the fixed reduction tree.
         */
        inline float reduce_gpu_aligned_expert_partials_avx2(
            float *partials) noexcept
        {
            constexpr int kLanes =
                floating_expert_numerical_contract::kReductionLanes;
            for (int stride = kLanes / 2; stride >= 8; stride >>= 1)
            {
                for (int lane = 0; lane < stride; lane += 8)
                {
                    const __m256 lhs = _mm256_load_ps(partials + lane);
                    const __m256 rhs =
                        _mm256_load_ps(partials + lane + stride);
                    _mm256_store_ps(
                        partials + lane,
                        _mm256_add_ps(lhs, rhs));
                }
            }
            for (int stride = 4; stride > 0; stride >>= 1)
            {
                for (int lane = 0; lane < stride; ++lane)
                {
                    partials[lane] = device_fp32_contract::add(
                        partials[lane], partials[lane + stride]);
                }
            }
            return partials[0];
        }

        /**
         * @brief Compute one output column with a latency-covering AVX2
         *        schedule while retaining the exact 256-lane device tree.
         *
         * AVX2 has half the vector register file available to the AVX-512
         * implementation, so it keeps eight accumulator chains live.  The
         * compile-time row count divides that budget between independent rows;
         * this avoids spills while preserving each lane's K accumulation order.
         *
         * @tparam WeightType Native floating tensor storage type.
         * @tparam Rows Compile-time row count in `[1,4]`.
         * @param activations First activation row for this tile.
         * @param weights Complete row-major transposed weight tensor.
         * @param weight_row Element offset of the selected output column.
         * @param k Reduction width and activation-row stride.
         * @param reduced Exact tree root for each row.
         */
        template <TensorType WeightType, int Rows>
        inline void dot_gpu_aligned_expert_rows_avx2(
            const float *activations,
            const void *weights,
            std::size_t weight_row,
            int k,
            float *reduced) noexcept
        {
            static_assert(Rows >= 1 && Rows <= 4);
            constexpr int kLanes =
                floating_expert_numerical_contract::kReductionLanes;
            constexpr int kVectorLanes = 8;
            constexpr int kLaneBlocksPerTile = 8 / Rows;
            alignas(32) float partials[Rows][kLanes] = {};

            for (int lane_tile = 0; lane_tile < kLanes;
                 lane_tile += kVectorLanes * kLaneBlocksPerTile)
            {
                __m256 accumulators[Rows][kLaneBlocksPerTile];
                // As above, unroll both fixed dimensions so the eight live
                // chains remain YMM registers rather than stack slots.
#pragma GCC unroll 4
                for (int row = 0; row < Rows; ++row)
                {
#pragma GCC unroll 8
                    for (int block = 0; block < kLaneBlocksPerTile; ++block)
                        accumulators[row][block] = _mm256_setzero_ps();
                }

                for (int base = 0; base < k; base += kLanes)
                {
#pragma GCC unroll 8
                    for (int block = 0; block < kLaneBlocksPerTile; ++block)
                    {
                        const int lane_block =
                            lane_tile + block * kVectorLanes;
                        if (lane_block >= kLanes)
                            continue;
                        const int first_k = base + lane_block;
                        if (first_k >= k)
                            continue;
                        const int valid = std::min(kVectorLanes, k - first_k);
                        const __m256 weight_values =
                            load_gpu_aligned_expert_weight_avx2<WeightType>(
                                weights,
                                weight_row +
                                    static_cast<std::size_t>(first_k),
                                valid);
#pragma GCC unroll 4
                        for (int row = 0; row < Rows; ++row)
                        {
                            const float *activation_values =
                                activations +
                                static_cast<std::size_t>(row) * k +
                                static_cast<std::size_t>(first_k);
                            accumulators[row][block] = _mm256_fmadd_ps(
                                load_gpu_aligned_expert_activation_avx2(
                                    activation_values, valid),
                                weight_values,
                                accumulators[row][block]);
                        }
                    }
                }

#pragma GCC unroll 4
                for (int row = 0; row < Rows; ++row)
                {
#pragma GCC unroll 8
                    for (int block = 0; block < kLaneBlocksPerTile; ++block)
                    {
                        const int lane_block =
                            lane_tile + block * kVectorLanes;
                        if (lane_block < kLanes)
                        {
                            _mm256_store_ps(
                                partials[row] + lane_block,
                                accumulators[row][block]);
                        }
                    }
                }
            }

            for (int row = 0; row < Rows; ++row)
            {
                reduced[row] =
                    reduce_gpu_aligned_expert_partials_avx2(partials[row]);
            }
        }
#endif

        /**
         * @brief Execute the fixed GPU-aligned expert FP32 dot-product program.
         *
         * GPU ExpertOverlay kernels reduce 256 strided K lanes through a fixed
         * binary tree. CPU-resident GPU-aligned experts emulate that exact program;
         * otherwise promotion changes the logical model and recursive MTP
         * amplifies the placement-dependent perturbation. AVX-512 computes 16
         * independent logical lanes at once. It never vector-reduces K, so its
         * result remains byte-identical to the scalar oracle and both GPU
         * backends. AVX2 computes eight independent lanes per instruction;
         * runtime dispatch uses the process-wide typed ISA authority.
         *
         * @tparam WeightType Native floating tensor storage type.
         * @param A Row-major FP32 activations `[M,K]`.
         * @param B Row-major transposed weights `[N,K]`.
         * @param C Row-major FP32 output `[M,N]`.
         * @param M Runtime row count.
         * @param N Output width.
         * @param K Reduction width.
         * @param alpha Output scale.
         * @param beta Existing-output scale.
         * @param bias Optional FP32 output-column bias.
         * @return True after every output word is published.
         */
        template <TensorType WeightType, ISALevel ISA>
        inline bool run_gpu_aligned_expert_fp32_matmul_isa(
            const float *A,
            const void *B,
            float *C,
            int M,
            int N,
            int K,
            float alpha = 1.0f,
            float beta = 0.0f,
            const float *bias = nullptr)
        {
            static_assert(
                WeightType == TensorType::FP32 ||
                    WeightType == TensorType::FP16 ||
                    WeightType == TensorType::BF16,
                "GPU-aligned expert weights must use a floating storage type");
            static_assert(
                ISA == ISALevel::Scalar || ISA == ISALevel::AVX2 ||
                    ISA == ISALevel::AVX512,
                "GPU-aligned expert matmul needs a concrete ISA implementation");
            if (!A || !B || !C || M < 0 || N < 0 || K <= 0)
            {
                LOG_ERROR("[FloatingPointGemmKernel] Invalid GPU-aligned expert matmul geometry"
                          << " A=" << static_cast<const void *>(A)
                          << " B=" << B
                          << " C=" << static_cast<void *>(C)
                          << " M=" << M << " N=" << N << " K=" << K);
                return false;
            }

            constexpr int kRowTile = 4;
            constexpr int kLanes =
                floating_expert_numerical_contract::kReductionLanes;
            static_assert(kLanes == 256,
                          "CPU SIMD schedules implement the 256-lane device tree");
            const int row_tiles = (M + kRowTile - 1) / kRowTile;
            auto work = [&]()
            {
#pragma omp for collapse(2) schedule(static)
                for (int row_tile = 0; row_tile < row_tiles; ++row_tile)
                {
                    for (int column = 0; column < N; ++column)
                    {
                        const int first_row = row_tile * kRowTile;
                        const int tile_rows =
                            std::min(kRowTile, M - first_row);
                        std::array<float, kRowTile> reduced{};
#if defined(__AVX512F__) && defined(__AVX512BW__)
                        if constexpr (ISA == ISALevel::AVX512)
                        {
                            const std::size_t weight_row =
                                static_cast<std::size_t>(column) * K;
                            const float *tile_activations =
                                A + static_cast<std::size_t>(first_row) * K;
                            // Specializing the small runtime row tail lets the
                            // scheduler keep several independent logical-lane
                            // chains live without changing their arithmetic.
                            switch (tile_rows)
                            {
                                case 1:
                                    dot_gpu_aligned_expert_rows_avx512<
                                        WeightType,
                                        1>(tile_activations,
                                           B,
                                           weight_row,
                                           K,
                                           reduced.data());
                                    break;
                                case 2:
                                    dot_gpu_aligned_expert_rows_avx512<
                                        WeightType,
                                        2>(tile_activations,
                                           B,
                                           weight_row,
                                           K,
                                           reduced.data());
                                    break;
                                case 3:
                                    dot_gpu_aligned_expert_rows_avx512<
                                        WeightType,
                                        3>(tile_activations,
                                           B,
                                           weight_row,
                                           K,
                                           reduced.data());
                                    break;
                                case 4:
                                    dot_gpu_aligned_expert_rows_avx512<
                                        WeightType,
                                        4>(tile_activations,
                                           B,
                                           weight_row,
                                           K,
                                           reduced.data());
                                    break;
                            }
                        }
                        else
#endif
#if defined(__AVX2__)
                        if constexpr (ISA == ISALevel::AVX2)
                        {
                            const std::size_t weight_row =
                                static_cast<std::size_t>(column) * K;
                            const float *tile_activations =
                                A + static_cast<std::size_t>(first_row) * K;
                            switch (tile_rows)
                            {
                                case 1:
                                    dot_gpu_aligned_expert_rows_avx2<
                                        WeightType,
                                        1>(tile_activations,
                                           B,
                                           weight_row,
                                           K,
                                           reduced.data());
                                    break;
                                case 2:
                                    dot_gpu_aligned_expert_rows_avx2<
                                        WeightType,
                                        2>(tile_activations,
                                           B,
                                           weight_row,
                                           K,
                                           reduced.data());
                                    break;
                                case 3:
                                    dot_gpu_aligned_expert_rows_avx2<
                                        WeightType,
                                        3>(tile_activations,
                                           B,
                                           weight_row,
                                           K,
                                           reduced.data());
                                    break;
                                case 4:
                                    dot_gpu_aligned_expert_rows_avx2<
                                        WeightType,
                                        4>(tile_activations,
                                           B,
                                           weight_row,
                                           K,
                                           reduced.data());
                                    break;
                            }
                        }
                        else
#endif
                        {
                            floating_expert_numerical_contract::dotRows<
                                kRowTile>(
                                tile_rows,
                                K,
                                [&](int tile_row, int k)
                                {
                                    return A[
                                        static_cast<std::size_t>(
                                            first_row + tile_row) * K +
                                        static_cast<std::size_t>(k)];
                                },
                                [&](int k)
                                {
                                    return load_gpu_aligned_expert_weight_scalar<
                                        WeightType>(
                                        B,
                                        static_cast<std::size_t>(column) * K +
                                            static_cast<std::size_t>(k));
                                },
                                reduced);
                        }

                        const float bias_value = bias ? bias[column] : 0.0f;
                        for (int tile_row = 0; tile_row < tile_rows;
                             ++tile_row)
                        {
                            const std::size_t output_index =
                                static_cast<std::size_t>(
                                    first_row + tile_row) * N +
                                static_cast<std::size_t>(column);
                            float value = alpha * reduced[
                                static_cast<std::size_t>(tile_row)];
                            value += bias_value;
                            if (beta != 0.0f)
                                value += beta * C[output_index];
                            C[output_index] = value;
                        }
                    }
                }
            };
            OMP_WORKSHARE_REGION(work);
            return true;
        }

        /**
         * @brief Execute the portable scalar GPU-aligned expert schedule.
         * @tparam WeightType Native floating tensor storage type.
         */
        template <TensorType WeightType>
        inline bool run_gpu_aligned_expert_fp32_matmul_scalar(
            const float *A,
            const void *B,
            float *C,
            int M,
            int N,
            int K,
            float alpha = 1.0f,
            float beta = 0.0f,
            const float *bias = nullptr)
        {
            return run_gpu_aligned_expert_fp32_matmul_isa<
                WeightType, ISALevel::Scalar>(
                A, B, C, M, N, K, alpha, beta, bias);
        }

#if defined(__AVX2__)
        /**
         * @brief Execute the AVX2 GPU-aligned expert schedule.
         * @tparam WeightType Native floating tensor storage type.
         */
        template <TensorType WeightType>
        inline bool run_gpu_aligned_expert_fp32_matmul_avx2(
            const float *A,
            const void *B,
            float *C,
            int M,
            int N,
            int K,
            float alpha = 1.0f,
            float beta = 0.0f,
            const float *bias = nullptr)
        {
            return run_gpu_aligned_expert_fp32_matmul_isa<
                WeightType, ISALevel::AVX2>(
                A, B, C, M, N, K, alpha, beta, bias);
        }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
        /**
         * @brief Execute the AVX-512 GPU-aligned expert schedule.
         * @tparam WeightType Native floating tensor storage type.
         */
        template <TensorType WeightType>
        inline bool run_gpu_aligned_expert_fp32_matmul_avx512(
            const float *A,
            const void *B,
            float *C,
            int M,
            int N,
            int K,
            float alpha = 1.0f,
            float beta = 0.0f,
            const float *bias = nullptr)
        {
            return run_gpu_aligned_expert_fp32_matmul_isa<
                WeightType, ISALevel::AVX512>(
                A, B, C, M, N, K, alpha, beta, bias);
        }
#endif

        /**
         * @brief Dispatch once to the process-selected GPU-aligned expert ISA.
         *
         * Dispatch happens before opening the OpenMP team, rather than once per
         * output column. This is the repository's scalar/AVX2/AVX-512 pattern:
         * the build controls the maximum compiled ISA and `activeISALevel()`
         * selects one complete implementation for the process.
         *
         * @tparam WeightType Native floating tensor storage type.
         * @return True after the selected exact-tree implementation completes.
         */
        template <TensorType WeightType>
        inline bool run_gpu_aligned_expert_fp32_matmul(
            const float *A,
            const void *B,
            float *C,
            int M,
            int N,
            int K,
            float alpha = 1.0f,
            float beta = 0.0f,
            const float *bias = nullptr)
        {
            switch (activeISALevel())
            {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            case ISALevel::AVX512:
                return run_gpu_aligned_expert_fp32_matmul_avx512<WeightType>(
                    A, B, C, M, N, K, alpha, beta, bias);
#endif
#if defined(__AVX2__)
            case ISALevel::AVX2:
                return run_gpu_aligned_expert_fp32_matmul_avx2<WeightType>(
                    A, B, C, M, N, K, alpha, beta, bias);
#endif
            case ISALevel::Scalar:
            default:
                return run_gpu_aligned_expert_fp32_matmul_scalar<WeightType>(
                    A, B, C, M, N, K, alpha, beta, bias);
            }
        }

        /**
         * @brief Execute FP32 matrix multiplication using OneDNN with optional fused bias
         *
         * @param A Input matrix A [M, K] (FP32, row-major)
         * @param B Input matrix B [K, N] or [N, K] if transpose_B (FP32)
         * @param C Output matrix C [M, N] (FP32, row-major)
         * @param M Number of rows in A and C
         * @param N Number of columns in B and C
         * @param K Number of columns in A and rows in B
         * @param transpose_B Whether B is transposed (stored as [N, K])
         * @param alpha Scale factor for A*B
         * @param beta Scale factor for existing C (for accumulation)
         * @param bias Optional bias vector [N] to fuse into GEMM (nullptr = no bias)
         * @return true on success
         */
        inline bool run_onednn_fp32_matmul(const float *A,
                                           const float *B,
                                           float *C,
                                           int M,
                                           int N,
                                           int K,
                                           bool transpose_B,
                                           float alpha = 1.0f,
                                           float beta = 0.0f,
                                           const float *bias = nullptr)
        {
            const KernelType profile_type = (M == 1) ? KernelType::GEMV_FP32 : KernelType::GEMM_FP32;
            KERNEL_PROFILE_SCOPE(profile_type);

            if (!A || !B || !C)
            {
                LOG_ERROR("OneDNN FP32 matmul received null pointer: "
                          << "A=" << static_cast<const void *>(A)
                          << " B=" << static_cast<const void *>(B)
                          << " C=" << static_cast<void *>(C)
                          << " M=" << M << " N=" << N << " K=" << K);
                return false;
            }

            /*
             * Public M=1 decode is the arithmetic oracle for MTP verifier
             * publication, so transposed-weight decode uses the local skinny
             * primitive. Dedicated grouped-verifier entrypoints below invoke
             * the same row-tiled primitive explicitly for every runtime M.
             * Ordinary M>1 GEMM remains on oneDNN; inferring verifier semantics
             * from a historical `M<=4` size range would conflate execution mode
             * with shape and break as soon as speculative depth grows.
             */
            if (B && transpose_B && M == 1)
            {
                return run_fp32_skinny_matmul(A, B, C, M, N, K, transpose_B, alpha, beta, bias);
            }

            if (B && M <= 64 && N <= 64)
            {
                return run_fp32_skinny_matmul(A, B, C, M, N, K, transpose_B, alpha, beta, bias);
            }

            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::f32, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f32, transpose_B ? tag::ba : tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);

                // Optional bias descriptor - shape [1, N] for broadcast across rows
                dnnl::memory::desc bias_md;
                if (bias)
                {
                    dnnl::memory::dims bias_dims = {1, N};
                    bias_md = dnnl::memory::desc(bias_dims, dt::f32, tag::ab);
                }

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Use eltwise linear for alpha scaling: f(x) = alpha * x + 0
                // This is more robust than set_scales_mask for simple scalar multiplication
                if (alpha != 1.0f)
                {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
                }
                if (beta != 0.0f)
                {
                    ops.append_sum(beta);
                }
                attr.set_post_ops(ops);

                // Create primitive descriptor with or without bias
                dnnl::matmul::primitive_desc matmul_pd = bias
                                                             ? dnnl::matmul::primitive_desc(onednn_engine(), src_md, weight_md, bias_md, dst_md, attr)
                                                             : dnnl::matmul::primitive_desc(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<float *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<float *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                std::unordered_map<int, dnnl::memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weight_mem});
                args.insert({DNNL_ARG_DST, dst_mem});

                // Add bias memory if provided
                if (bias)
                {
                    dnnl::memory bias_mem(bias_md, onednn_engine(), const_cast<float *>(bias));
                    args.insert({DNNL_ARG_BIAS, bias_mem});
                }

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_ERROR("OneDNN FP32 matmul failed: status=" << e.status << " message=" << e.what());
                return false;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("OneDNN FP32 matmul failed: " << e.what());
                return false;
            }

            return true;
        }

        /**
         * @brief Execute BF16 matrix multiplication using OneDNN (bf16×bf16→f32)
         *
         * @param A Input matrix A [M, K] (BF16 as uint16_t, row-major)
         * @param B Input matrix B [K, N] (BF16 as uint16_t, row-major)
         * @param C Output matrix C [M, N] (FP32, row-major)
         * @param M Number of rows in A and C
         * @param N Number of columns in B and C
         * @param K Number of columns in A and rows in B
         * @param transpose_B Whether B is transposed
         * @param alpha Scale factor for A*B
         * @param beta Scale factor for existing C
         * @return true on success
         */
        inline bool run_onednn_bf16_matmul(const uint16_t *A,
                                           const uint16_t *B,
                                           float *C,
                                           int M,
                                           int N,
                                           int K,
                                           bool transpose_B = false,
                                           float alpha = 1.0f,
                                           float beta = 0.0f)
        {
            const KernelType profile_type = (M == 1) ? KernelType::GEMV_FP32 : KernelType::GEMM_FP32;
            KERNEL_PROFILE_SCOPE(profile_type);

            if (B && transpose_B && M == 1)
            {
                return run_bf16_skinny_matmul(A, B, C, M, N, K, transpose_B, alpha, beta);
            }

            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::bf16, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::bf16, transpose_B ? tag::ba : tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Use eltwise linear for alpha scaling: f(x) = alpha * x + 0
                if (alpha != 1.0f)
                {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
                }
                if (beta != 0.0f)
                {
                    ops.append_sum(beta);
                }
                attr.set_post_ops(ops);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<uint16_t *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<uint16_t *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                std::unordered_map<int, dnnl::memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weight_mem});
                args.insert({DNNL_ARG_DST, dst_mem});

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_ERROR("OneDNN BF16 matmul failed: status=" << e.status << " message=" << e.what());
                return false;
            }

            return true;
        }

        /**
         * @brief Execute FP16 matrix multiplication using OneDNN (fp16×fp16→f32)
         *
         * @param A Input matrix A [M, K] (FP16 as uint16_t, row-major)
         * @param B Input matrix B [K, N] (FP16 as uint16_t, row-major)
         * @param C Output matrix C [M, N] (FP32, row-major)
         * @param M Number of rows in A and C
         * @param N Number of columns in B and C
         * @param K Number of columns in A and rows in B
         * @param transpose_B Whether B is transposed
         * @param alpha Scale factor for A*B
         * @param beta Scale factor for existing C
         * @return true on success
         */
        inline bool run_onednn_fp16_matmul(const uint16_t *A,
                                           const uint16_t *B,
                                           float *C,
                                           int M,
                                           int N,
                                           int K,
                                           bool transpose_B = false,
                                           float alpha = 1.0f,
                                           float beta = 0.0f)
        {
            const KernelType profile_type = (M == 1) ? KernelType::GEMV_FP32 : KernelType::GEMM_FP32;
            KERNEL_PROFILE_SCOPE(profile_type);

            if (B && transpose_B && M == 1)
            {
                return run_fp16_skinny_matmul(A, B, C, M, N, K, transpose_B, alpha, beta);
            }

            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::f16, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f16, transpose_B ? tag::ba : tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Use eltwise linear for alpha scaling: f(x) = alpha * x + 0
                if (alpha != 1.0f)
                {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
                }
                if (beta != 0.0f)
                {
                    ops.append_sum(beta);
                }
                attr.set_post_ops(ops);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<uint16_t *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<uint16_t *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                std::unordered_map<int, dnnl::memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weight_mem});
                args.insert({DNNL_ARG_DST, dst_mem});

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_WARN("OneDNN FP16 matmul failed (likely unsupported hardware): status=" << e.status
                                                                                            << " message=" << e.what());
                return false;
            }

            return true;
        }

        // ========== Strided OneDNN GEMM Primitives ==========

        /**
         * @brief Execute FP32 strided matrix multiplication using OneDNN
         *
         * Supports non-contiguous memory layouts via custom strides.
         * Essential for multi-head attention where Q/K/V heads are interleaved.
         *
         * @param A Input matrix A [M, K] with leading dimension lda
         * @param B Input matrix B [N, K] if transpose_B, else [K, N], with leading dimension ldb
         * @param C Output matrix C [M, N] with leading dimension ldc
         * @param M Number of rows in A and C
         * @param N Number of rows in B (if transpose_B) or columns in B
         * @param K Number of columns in A
         * @param lda Leading dimension of A (stride between rows)
         * @param ldb Leading dimension of B (stride between rows)
         * @param ldc Leading dimension of C (stride between rows)
         * @param transpose_B Whether B is transposed (stored as [N, K])
         * @param alpha Scale factor for A*B
         * @param beta Scale factor for existing C (for accumulation)
         * @return true on success
         */
        inline bool run_onednn_fp32_matmul_strided(const float *A,
                                                   const float *B,
                                                   float *C,
                                                   int M,
                                                   int N,
                                                   int K,
                                                   int lda,
                                                   int ldb,
                                                   int ldc,
                                                   bool transpose_B,
                                                   float alpha = 1.0f,
                                                   float beta = 0.0f)
        {
            const KernelType profile_type = (M == 1) ? KernelType::GEMV_FP32 : KernelType::GEMM_FP32;
            KERNEL_PROFILE_SCOPE(profile_type);

            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                // Memory dimensions (logical shape)
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims dst_dims = {M, N};

                // Strides for row-major layout: {stride_between_rows, stride_between_cols}
                // For row-major: stride_between_cols = 1, stride_between_rows = lda
                dnnl::memory::dims src_strides = {lda, 1};
                dnnl::memory::dims dst_strides = {ldc, 1};

                // Weight dimensions and strides depend on transpose_B
                dnnl::memory::dims weight_dims;
                dnnl::memory::dims weight_strides;

                if (transpose_B)
                {
                    // B is stored as [N, K] (each row is a K-vector)
                    // We want to compute A @ B^T, so logical weight shape is [K, N]
                    weight_dims = {K, N};
                    // B^T: to get element at logical [k, n], we access B[n, k] = B[n * ldb + k]
                    // OneDNN strides for transposed: {1, ldb} means:
                    //   - stride along K (first dim) = 1
                    //   - stride along N (second dim) = ldb
                    weight_strides = {1, ldb};
                }
                else
                {
                    // B is stored as [K, N], logical shape is [K, N]
                    weight_dims = {K, N};
                    weight_strides = {ldb, 1};
                }

                auto src_md = dnnl::memory::desc(src_dims, dt::f32, src_strides);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f32, weight_strides);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, dst_strides);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Use eltwise linear for alpha scaling: f(x) = alpha * x + 0
                // This is more robust than set_scales_mask for simple scalar multiplication
                if (alpha != 1.0f)
                {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
                }

                if (beta != 0.0f)
                {
                    ops.append_sum(beta);
                }
                attr.set_post_ops(ops);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<float *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<float *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                std::unordered_map<int, dnnl::memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weight_mem});
                args.insert({DNNL_ARG_DST, dst_mem});

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_ERROR("OneDNN FP32 strided matmul failed: status=" << e.status << " message=" << e.what());
                return false;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("OneDNN FP32 strided matmul failed: " << e.what());
                return false;
            }

            return true;
        }

        /**
         * @brief Execute BF16 strided matrix multiplication using OneDNN
         */
        inline bool run_onednn_bf16_matmul_strided(const uint16_t *A,
                                                   const uint16_t *B,
                                                   float *C,
                                                   int M,
                                                   int N,
                                                   int K,
                                                   int lda,
                                                   int ldb,
                                                   int ldc,
                                                   bool transpose_B,
                                                   float alpha = 1.0f,
                                                   float beta = 0.0f)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims dst_dims = {M, N};
                dnnl::memory::dims src_strides = {lda, 1};
                dnnl::memory::dims dst_strides = {ldc, 1};

                dnnl::memory::dims weight_dims;
                dnnl::memory::dims weight_strides;

                if (transpose_B)
                {
                    weight_dims = {K, N};
                    weight_strides = {1, ldb};
                }
                else
                {
                    weight_dims = {K, N};
                    weight_strides = {ldb, 1};
                }

                auto src_md = dnnl::memory::desc(src_dims, dt::bf16, src_strides);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::bf16, weight_strides);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, dst_strides);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Use eltwise linear for alpha scaling
                if (alpha != 1.0f)
                {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
                }
                if (beta != 0.0f)
                {
                    ops.append_sum(beta);
                }
                attr.set_post_ops(ops);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<uint16_t *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<uint16_t *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                std::unordered_map<int, dnnl::memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weight_mem});
                args.insert({DNNL_ARG_DST, dst_mem});

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_ERROR("OneDNN BF16 strided matmul failed: status=" << e.status << " message=" << e.what());
                return false;
            }

            return true;
        }

        /**
         * @brief Execute FP16 strided matrix multiplication using OneDNN
         */
        inline bool run_onednn_fp16_matmul_strided(const uint16_t *A,
                                                   const uint16_t *B,
                                                   float *C,
                                                   int M,
                                                   int N,
                                                   int K,
                                                   int lda,
                                                   int ldb,
                                                   int ldc,
                                                   bool transpose_B,
                                                   float alpha = 1.0f,
                                                   float beta = 0.0f)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims dst_dims = {M, N};
                dnnl::memory::dims src_strides = {lda, 1};
                dnnl::memory::dims dst_strides = {ldc, 1};

                dnnl::memory::dims weight_dims;
                dnnl::memory::dims weight_strides;

                if (transpose_B)
                {
                    weight_dims = {K, N};
                    weight_strides = {1, ldb};
                }
                else
                {
                    weight_dims = {K, N};
                    weight_strides = {ldb, 1};
                }

                auto src_md = dnnl::memory::desc(src_dims, dt::f16, src_strides);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f16, weight_strides);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, dst_strides);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Use eltwise linear for alpha scaling
                if (alpha != 1.0f)
                {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
                }
                if (beta != 0.0f)
                {
                    ops.append_sum(beta);
                }
                attr.set_post_ops(ops);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<uint16_t *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<uint16_t *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                std::unordered_map<int, dnnl::memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weight_mem});
                args.insert({DNNL_ARG_DST, dst_mem});

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_WARN("OneDNN FP16 strided matmul failed: status=" << e.status << " message=" << e.what());
                return false;
            }

            return true;
        }

        /**
         * @brief Execute mixed FP32×BF16 strided matrix multiplication using OneDNN
         *
         * Used for attention: FP32 scores × BF16 V → FP32 output
         */
        inline bool run_onednn_fp32_bf16_matmul_strided(const float *A,
                                                        const uint16_t *B,
                                                        float *C,
                                                        int M,
                                                        int N,
                                                        int K,
                                                        int lda,
                                                        int ldb,
                                                        int ldc,
                                                        bool transpose_B,
                                                        float alpha = 1.0f,
                                                        float beta = 0.0f)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims dst_dims = {M, N};
                dnnl::memory::dims src_strides = {lda, 1};
                dnnl::memory::dims dst_strides = {ldc, 1};

                dnnl::memory::dims weight_dims;
                dnnl::memory::dims weight_strides;

                if (transpose_B)
                {
                    weight_dims = {K, N};
                    weight_strides = {1, ldb};
                }
                else
                {
                    weight_dims = {K, N};
                    weight_strides = {ldb, 1};
                }

                // Mixed precision: FP32 source, BF16 weights, FP32 output
                auto src_md = dnnl::memory::desc(src_dims, dt::f32, src_strides);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::bf16, weight_strides);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, dst_strides);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Use eltwise linear for alpha scaling
                if (alpha != 1.0f)
                {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
                }
                if (beta != 0.0f)
                {
                    ops.append_sum(beta);
                }
                attr.set_post_ops(ops);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<float *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<uint16_t *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                std::unordered_map<int, dnnl::memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weight_mem});
                args.insert({DNNL_ARG_DST, dst_mem});

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_WARN("OneDNN FP32×BF16 strided matmul failed: status=" << e.status << " message=" << e.what());
                return false;
            }

            return true;
        }

        /**
         * @brief Execute mixed FP32×FP16 strided matrix multiplication using OneDNN
         *
         * Used for attention: FP32 scores × FP16 V → FP32 output
         */
        inline bool run_onednn_fp32_fp16_matmul_strided(const float *A,
                                                        const uint16_t *B,
                                                        float *C,
                                                        int M,
                                                        int N,
                                                        int K,
                                                        int lda,
                                                        int ldb,
                                                        int ldc,
                                                        bool transpose_B,
                                                        float alpha = 1.0f,
                                                        float beta = 0.0f)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims dst_dims = {M, N};
                dnnl::memory::dims src_strides = {lda, 1};
                dnnl::memory::dims dst_strides = {ldc, 1};

                dnnl::memory::dims weight_dims;
                dnnl::memory::dims weight_strides;

                if (transpose_B)
                {
                    weight_dims = {K, N};
                    weight_strides = {1, ldb};
                }
                else
                {
                    weight_dims = {K, N};
                    weight_strides = {ldb, 1};
                }

                // Mixed precision: FP32 source, FP16 weights, FP32 output
                auto src_md = dnnl::memory::desc(src_dims, dt::f32, src_strides);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f16, weight_strides);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, dst_strides);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Use eltwise linear for alpha scaling
                if (alpha != 1.0f)
                {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
                }
                if (beta != 0.0f)
                {
                    ops.append_sum(beta);
                }
                attr.set_post_ops(ops);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<float *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<uint16_t *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                std::unordered_map<int, dnnl::memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weight_mem});
                args.insert({DNNL_ARG_DST, dst_mem});

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_WARN("OneDNN FP32×FP16 strided matmul failed: status=" << e.status << " message=" << e.what());
                return false;
            }

            return true;
        }

        /**
         * @brief Execute Softmax using OneDNN
         *
         * Computes y = softmax(x) along the last dimension (axis=1).
         * Supports in-place operation (src == dst).
         */
        inline void run_onednn_softmax(float *data, int rows, int cols, int stride)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                // Logical dimensions: {rows, cols}
                dnnl::memory::dims dims = {rows, cols};

                // Strides: {stride, 1}
                // OneDNN expects strides to match the memory layout
                dnnl::memory::dims strides = {stride, 1};

                auto md = dnnl::memory::desc(dims, dt::f32, strides);

                // Create memory object wrapping the data
                dnnl::memory mem(md, onednn_engine(), data);

                // Softmax primitive descriptor
                // axis = 1 (columns)
                auto softmax_pd = dnnl::softmax_forward::primitive_desc(
                    onednn_engine(),
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::softmax_accurate,
                    md,
                    md,
                    1);

                // Execute
                dnnl::softmax_forward(softmax_pd).execute(onednn_stream(), {
                                                                               {DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem} // In-place
                                                                           });

                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_ERROR("OneDNN softmax failed: status=" << e.status << " message=" << e.what());
            }
        }

        // ========== Floating-Point GEMM Kernel ==========

        /**
         * @brief ITensorGemm implementation for floating-point GEMM
         *
         * Supports only homogeneous floating-point type combinations:
         * - FP32 weights × FP32 activations
         * - FP16 weights × FP16 activations
         * - BF16 weights × BF16 activations
         *
         * For quantized weight GEMM, use CPUQuantisedGemmKernel instead.
         */
        class FloatingPointGemmKernel : public ITensorGemm, public CPUKernelBase
        {
        private:
            /**
             * @brief Copy one expert view into engine-owned execution storage.
             *
             * A prepared expert may outlive the graph-frozen loader tensor and
             * may later become a recyclable ExpertOverlay physical slot. Keeping
             * a shared pointer to a view is insufficient because the loader is
             * allowed to retire the parent's native vector after preparation.
             * This method preserves the native bytes exactly—no conversion or
             * arithmetic occurs—and returns an owning, contiguous tensor whose
             * lifetime is identical to the GEMM engine's lifetime.
             *
             * @param source Live FP32, FP16, or BF16 expert matrix.
             * @param placement Final native storage placement before copying source bytes.
             * @return Independent owning tensor containing the exact bytes.
             * @throws std::invalid_argument for null, non-matrix, unsupported,
             *         released, or internally inconsistent source storage.
             */
            static std::shared_ptr<const TensorBase>
            cloneExpertExecutionStorage(
                const std::shared_ptr<const TensorBase> &source,
                CPUWeightStoragePlacement placement)
            {
                if (!source || source->shape().size() != 2u ||
                    !source->raw_data())
                {
                    throw std::invalid_argument(
                        "Prepared floating expert requires a live 2-D source tensor");
                }

                std::shared_ptr<TensorBase> owned;
                // Allocate the final native storage before its first write;
                // tensor adoption preserves the page mapping without a copy.
                const auto copy_storage = [&]<typename T>() {
                    if (source->size_bytes() % sizeof(T) != 0)
                        throw std::invalid_argument("Floating expert byte count is not element-aligned");
                    auto storage = placement.allocate<T>(source->size_bytes() / sizeof(T));
                    std::memcpy(storage.data(), source->raw_data(), source->size_bytes());
                    return storage;
                };
                switch (source->native_type())
                {
                case TensorType::FP32:
                    owned = std::make_shared<FP32Tensor>(source->shape(), copy_storage.template operator()<float>());
                    break;
                case TensorType::FP16:
                    owned = std::make_shared<FP16Tensor>(source->shape(), copy_storage.template operator()<uint16_t>());
                    break;
                case TensorType::BF16:
                    owned = std::make_shared<BF16Tensor>(source->shape(), copy_storage.template operator()<uint16_t>());
                    break;
                default:
                    throw std::invalid_argument(
                        "Prepared floating expert supports only FP32, FP16, and BF16");
                }

                if (!owned->raw_mutable_data() ||
                    owned->size_bytes() != source->size_bytes())
                {
                    throw std::logic_error(
                        "Prepared floating expert allocation does not match source storage");
                }

                // No conversion or second copy: FP16/BF16 native bits remain exact.
                return owned;
            }

        public:
            /**
             * @brief Shared arithmetic identity for prepared expert engines.
             *
             * Keeping this compatibility name avoids duplicating an enum: the
             * quantized and floating engines consume the same typed policy.
             */
            using NumericalPolicy = CPUProjectionNumericalPolicy;

            /**
             * @brief Construct a kernel that borrows a floating-point weight tensor.
             *
             * The caller must keep `weight_tensor` alive for the complete kernel
             * lifetime. Long-lived prepared-expert registries must use the
             * shared-ownership overload below instead.
             *
             * @param weight_tensor Borrowed weight tensor (FP32, FP16, or BF16).
             * @param numerical_policy Backend-native or GPU-aligned expert arithmetic.
             */
            explicit FloatingPointGemmKernel(
                const TensorBase *weight_tensor,
                NumericalPolicy numerical_policy = NumericalPolicy::BackendNative)
                : weight_tensor_(weight_tensor),
                  numerical_policy_(numerical_policy)
            {
                validateBoundWeight();
            }

            /**
             * @brief Construct a kernel with engine-owned expert weight bytes.
             *
             * The native bytes are copied exactly into a contiguous owning tensor.
             * This mirrors packed CPU and GPU prepared-engine semantics: once the
             * constructor returns, loader/source storage is no longer an input to
             * inference and may be reclaimed independently.
             *
             * @param weight_tensor Shared weight tensor (FP32, FP16, or BF16).
             * @param numerical_policy Backend-native or GPU-aligned expert arithmetic.
             * @param placement Final CPU storage placement, certified before the native copy.
             */
            explicit FloatingPointGemmKernel(
                std::shared_ptr<const TensorBase> weight_tensor,
                NumericalPolicy numerical_policy = NumericalPolicy::BackendNative,
                CPUWeightStoragePlacement placement = CPUWeightStoragePlacement::local())
                : weight_tensor_lifetime_(
                      cloneExpertExecutionStorage(weight_tensor, placement)),
                  weight_tensor_(weight_tensor_lifetime_.get()),
                  numerical_policy_(numerical_policy)
            {
                validateBoundWeight();
            }

            ~FloatingPointGemmKernel() override = default;

            /** @brief Export the engine's exact live row-major CPU weights. */
            bool exportContiguousFloatingPointWeights(
                ContiguousFloatingPointWeightDescriptor &out) const override
            {
                out = {};
                if (!weight_tensor_)
                    return false;
                out = {
                    .data = weight_tensor_->raw_data(),
                    .type = weight_type_,
                    .n = static_cast<int>(weight_tensor_->rows()),
                    .k = static_cast<int>(weight_tensor_->cols()),
                    .bytes = weight_tensor_->size_bytes(),
                };
                return out.valid();
            }

            /** @brief Expose a retired slot's final row-major storage. */
            std::span<std::uint8_t>
            exportRetiredCPUFloatingPointStorage() noexcept override
            {
                if (!weight_tensor_)
                    return {};
                void *bytes =
                    const_cast<TensorBase *>(weight_tensor_)->raw_mutable_data();
                return bytes
                           ? std::span<std::uint8_t>(
                                 static_cast<std::uint8_t *>(bytes),
                                 weight_tensor_->size_bytes())
                           : std::span<std::uint8_t>{};
            }

            /**
             * @brief Confirm that expert preparation detached from loader bytes.
             * @return Always true for the shared expert constructor's owning
             *         representation; borrowed dense callers never use the raw
             *         expert-preparation API.
             */
            bool canReleaseSourceWeightTensor() const override
            {
                return weight_tensor_lifetime_ != nullptr;
            }

            /**
             * @brief Check device support (CPU-only for OneDNN)
             */
            bool supports_device(int device_idx) const override
            {
                return device_idx == -1; // CPU only
            }

            // IKernelSnapshotCapable interface
            KernelSnapshotInfo getKernelSnapshotInfo() const override
            {
                // Determine weight dtype from bound tensor
                KernelBufferDtype weight_dtype = KernelBufferDtype::FP32;
                if (weight_type_ == TensorType::FP16)
                    weight_dtype = KernelBufferDtype::FP16;
                else if (weight_type_ == TensorType::BF16)
                    weight_dtype = KernelBufferDtype::BF16;

                return KernelSnapshotInfo::gemm()
                    .withInput("A", "input activations [m, k]", weight_dtype)
                    .withWeight("B", "weight matrix [n, k]", weight_dtype)
                    .withOutput("C", "output matrix [m, n]", weight_dtype)
                    .withScalar("m", "batch dimension", KernelBufferDtype::INT32)
                    .withScalar("n", "output features", KernelBufferDtype::INT32)
                    .withScalar("k", "input features", KernelBufferDtype::INT32)
                    .withScalar("alpha", "output scale factor")
                    .withScalar("beta", "accumulate scale factor");
            }

            // ========== Tensor-Based Interface: multiply_tensor() ==========

            /**
             * @brief Tensor-based GEMM with runtime type checking
             *
             * Validates the activation/weight pair before dispatch.
             * Supported combinations:
             * - FP32 activation × FP32 weight
             * - FP16 activation × FP16 weight
             * - BF16 activation × BF16 weight
             * - FP32 activation × FP16/BF16 weight
             *
             * @return true on success, false if type combination not supported
             */
            bool multiply_tensor(
                const TensorBase *A, TensorBase *C,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const TensorBase *bias = nullptr,
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr,
                int activation_row_offset = 0) override
            {
                (void)mpi_ctx;
                (void)device_idx;
                (void)workspace; // CPU kernel doesn't need external workspace

                if (!weight_tensor_ || !A || !C)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] Null tensor pointer");
                    return false;
                }

                const TensorType act_type = A->native_type();
                const auto &a_shape = A->shape();
                const auto &c_shape = C->shape();

                int m = static_cast<int>(a_shape[0]);
                int k = static_cast<int>(a_shape.size() > 1 ? a_shape[1] : 1);
                int n = static_cast<int>(c_shape.size() > 1 ? c_shape[1] : c_shape[0]);

                const bool mixed_fp32_activation =
                    act_type == TensorType::FP32 &&
                    (weight_type_ == TensorType::FP16 ||
                     weight_type_ == TensorType::BF16);
                if (act_type != weight_type_ && !mixed_fp32_activation)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] Unsupported activation/weight pair: activation="
                              << static_cast<int>(act_type)
                              << " weight=" << static_cast<int>(weight_type_));
                    return false;
                }

                // Extract bias pointer (FP32 fused bias only for now)
                const float *bias_ptr = bias ? bias->data() : nullptr;

                const size_t row_offset = static_cast<size_t>(activation_row_offset) * k;

                switch (weight_type_)
                {
                case TensorType::FP32:
                {
                    const float *A_data = A->data() + row_offset;
                    float *C_data = C->mutable_data();
                    const float *B_data = weight_tensor_->data();
                    if (numerical_policy_ == NumericalPolicy::GPUAlignedExpert)
                    {
                        if (!transpose_B)
                        {
                            LOG_ERROR("[FloatingPointGemmKernel] GPU-aligned floating experts require row-major transposed weights");
                            return false;
                        }
                        return run_gpu_aligned_expert_fp32_matmul<
                            TensorType::FP32>(
                            A_data,
                            B_data,
                            C_data,
                            m,
                            n,
                            k,
                            alpha,
                            beta,
                            bias_ptr);
                    }
                    // OneDNN FP32 matmul supports fused bias natively
                    return run_onednn_fp32_matmul(A_data, B_data, C_data, m, n, k, transpose_B, alpha, beta, bias_ptr);
                }

                case TensorType::FP16:
                {
                    const auto *B_fp16 = dynamic_cast<const FP16Tensor *>(weight_tensor_);
                    if (!B_fp16)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Failed to cast FP16 weights");
                        return false;
                    }
                    float *C_data = C->mutable_data();
                    if (bias_ptr)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Bias is unsupported for FP16 weights");
                        return false;
                    }
                    if (mixed_fp32_activation)
                    {
                        if (numerical_policy_ == NumericalPolicy::GPUAlignedExpert)
                        {
                            if (!transpose_B)
                            {
                                LOG_ERROR("[FloatingPointGemmKernel] GPU-aligned floating experts require row-major transposed weights");
                                return false;
                            }
                            return run_gpu_aligned_expert_fp32_matmul<
                                TensorType::FP16>(
                                A->data() + row_offset,
                                B_fp16->typed_data(),
                                C_data,
                                m,
                                n,
                                k,
                                alpha,
                                beta);
                        }
                        return run_fp32xfp16_skinny_matmul(
                            A->data() + row_offset,
                            B_fp16->typed_data(),
                            C_data,
                            m,
                            n,
                            k,
                            transpose_B,
                            alpha,
                            beta);
                    }
                    const auto *A_fp16 =
                        dynamic_cast<const FP16Tensor *>(A);
                    if (!A_fp16)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Failed to cast FP16 activations");
                        return false;
                    }
                    return run_onednn_fp16_matmul(
                        A_fp16->typed_data() + row_offset,
                        B_fp16->typed_data(),
                        C_data,
                        m,
                        n,
                        k,
                        transpose_B,
                        alpha,
                        beta);
                }

                case TensorType::BF16:
                {
                    const auto *B_bf16 = dynamic_cast<const BF16Tensor *>(weight_tensor_);
                    if (!B_bf16)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Failed to cast BF16 weights");
                        return false;
                    }
                    float *C_data = C->mutable_data();
                    if (bias_ptr)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Bias is unsupported for BF16 weights");
                        return false;
                    }
                    if (mixed_fp32_activation)
                    {
                        if (numerical_policy_ == NumericalPolicy::GPUAlignedExpert)
                        {
                            if (!transpose_B)
                            {
                                LOG_ERROR("[FloatingPointGemmKernel] GPU-aligned floating experts require row-major transposed weights");
                                return false;
                            }
                            return run_gpu_aligned_expert_fp32_matmul<
                                TensorType::BF16>(
                                A->data() + row_offset,
                                B_bf16->typed_data(),
                                C_data,
                                m,
                                n,
                                k,
                                alpha,
                                beta);
                        }
                        return run_fp32xbf16_skinny_matmul(
                            A->data() + row_offset,
                            B_bf16->typed_data(),
                            C_data,
                            m,
                            n,
                            k,
                            transpose_B,
                            alpha,
                            beta);
                    }
                    const auto *A_bf16 =
                        dynamic_cast<const BF16Tensor *>(A);
                    if (!A_bf16)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Failed to cast BF16 activations");
                        return false;
                    }
                    return run_onednn_bf16_matmul(
                        A_bf16->typed_data() + row_offset,
                        B_bf16->typed_data(),
                        C_data,
                        m,
                        n,
                        k,
                        transpose_B,
                        alpha,
                        beta);
                }

                default:
                    LOG_ERROR("[FloatingPointGemmKernel] Unsupported weight type: " << static_cast<int>(weight_type_));
                    return false;
                }
            }

            /**
             * @brief Tensor-based GEMM with explicit dimensions
             *
             * This overload uses caller-provided m, n, k instead of inferring from tensor shapes.
             * Essential for pre-allocated buffers where tensor shape > actual data size.
             *
             * @param A Input activations tensor [>=m, >=k]
             * @param C Output tensor [>=m, >=n]
             * @param m Number of rows to process
             * @param n Number of output columns
             * @param k Number of input columns
             * @param transpose_B Whether B is transposed (typical: true for weights)
             * @param alpha Scale factor
             * @param beta Accumulate factor
             * @param bias Optional bias tensor [n] to add after GEMM (nullptr = no bias)
             * @param mpi_ctx MPI context
             * @param device_idx Device index
             *
             * @return true on success
             */
            bool multiply_tensor(
                const TensorBase *A, TensorBase *C,
                int m, int n, int k,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const TensorBase *bias = nullptr,
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr,
                int activation_row_offset = 0) override
            {
                (void)mpi_ctx;
                (void)device_idx;
                (void)workspace; // CPU kernel doesn't need external workspace

                if (!weight_tensor_ || !A || !C)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] Null tensor pointer");
                    return false;
                }

                const TensorType act_type = A->native_type();

                const bool mixed_fp32_activation =
                    act_type == TensorType::FP32 &&
                    (weight_type_ == TensorType::FP16 ||
                     weight_type_ == TensorType::BF16);
                if (act_type != weight_type_ && !mixed_fp32_activation)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] Unsupported activation/weight pair: activation="
                              << static_cast<int>(act_type)
                              << " weight=" << static_cast<int>(weight_type_));
                    return false;
                }

                // Extract bias pointer (FP32 fused bias only for now)
                const float *bias_ptr = bias ? bias->data() : nullptr;

                const size_t row_offset = static_cast<size_t>(activation_row_offset) * k;

                switch (weight_type_)
                {
                case TensorType::FP32:
                {
                    const float *A_data = A->data() + row_offset;
                    float *C_data = C->mutable_data();
                    const float *B_data = weight_tensor_->data();
                    if (numerical_policy_ == NumericalPolicy::GPUAlignedExpert)
                    {
                        if (!transpose_B)
                        {
                            LOG_ERROR("[FloatingPointGemmKernel] GPU-aligned floating experts require row-major transposed weights");
                            return false;
                        }
                        return run_gpu_aligned_expert_fp32_matmul<
                            TensorType::FP32>(
                            A_data,
                            B_data,
                            C_data,
                            m,
                            n,
                            k,
                            alpha,
                            beta,
                            bias_ptr);
                    }
                    // OneDNN FP32 matmul supports fused bias natively
                    return run_onednn_fp32_matmul(A_data, B_data, C_data, m, n, k, transpose_B, alpha, beta, bias_ptr);
                }

                case TensorType::FP16:
                {
                    const auto *B_fp16 = dynamic_cast<const FP16Tensor *>(weight_tensor_);
                    if (!B_fp16)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Failed to cast FP16 weights");
                        return false;
                    }
                    float *C_data = C->mutable_data();
                    if (bias_ptr)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Bias is unsupported for FP16 weights");
                        return false;
                    }
                    if (mixed_fp32_activation)
                    {
                        if (numerical_policy_ == NumericalPolicy::GPUAlignedExpert)
                        {
                            if (!transpose_B)
                            {
                                LOG_ERROR("[FloatingPointGemmKernel] GPU-aligned floating experts require row-major transposed weights");
                                return false;
                            }
                            return run_gpu_aligned_expert_fp32_matmul<
                                TensorType::FP16>(
                                A->data() + row_offset,
                                B_fp16->typed_data(),
                                C_data,
                                m,
                                n,
                                k,
                                alpha,
                                beta);
                        }
                        return run_fp32xfp16_skinny_matmul(
                            A->data() + row_offset,
                            B_fp16->typed_data(),
                            C_data,
                            m,
                            n,
                            k,
                            transpose_B,
                            alpha,
                            beta);
                    }
                    const auto *A_fp16 =
                        dynamic_cast<const FP16Tensor *>(A);
                    if (!A_fp16)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Failed to cast FP16 activations");
                        return false;
                    }
                    return run_onednn_fp16_matmul(
                        A_fp16->typed_data() + row_offset,
                        B_fp16->typed_data(),
                        C_data,
                        m,
                        n,
                        k,
                        transpose_B,
                        alpha,
                        beta);
                }

                case TensorType::BF16:
                {
                    const auto *B_bf16 = dynamic_cast<const BF16Tensor *>(weight_tensor_);
                    if (!B_bf16)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Failed to cast BF16 weights");
                        return false;
                    }
                    float *C_data = C->mutable_data();
                    if (bias_ptr)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Bias is unsupported for BF16 weights");
                        return false;
                    }
                    if (mixed_fp32_activation)
                    {
                        if (numerical_policy_ == NumericalPolicy::GPUAlignedExpert)
                        {
                            if (!transpose_B)
                            {
                                LOG_ERROR("[FloatingPointGemmKernel] GPU-aligned floating experts require row-major transposed weights");
                                return false;
                            }
                            return run_gpu_aligned_expert_fp32_matmul<
                                TensorType::BF16>(
                                A->data() + row_offset,
                                B_bf16->typed_data(),
                                C_data,
                                m,
                                n,
                                k,
                                alpha,
                                beta);
                        }
                        return run_fp32xbf16_skinny_matmul(
                            A->data() + row_offset,
                            B_bf16->typed_data(),
                            C_data,
                            m,
                            n,
                            k,
                            transpose_B,
                            alpha,
                            beta);
                    }
                    const auto *A_bf16 =
                        dynamic_cast<const BF16Tensor *>(A);
                    if (!A_bf16)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Failed to cast BF16 activations");
                        return false;
                    }
                    return run_onednn_bf16_matmul(
                        A_bf16->typed_data() + row_offset,
                        B_bf16->typed_data(),
                        C_data,
                        m,
                        n,
                        k,
                        transpose_B,
                        alpha,
                        beta);
                }

                default:
                    LOG_ERROR("[FloatingPointGemmKernel] Unsupported weight type: " << static_cast<int>(weight_type_));
                    return false;
                }
            }

            /**
             * @brief Execute the ordinary CPU floating-point SwiGLU/down path.
             *
             * Shared experts produce FP32 gate and up activations regardless of
             * the stored down-projection format. The generic ITensorGemm
             * fallback cannot express that mixed-input contract, so returning
             * false here used to make every ordinary FP32 shared-expert stage
             * fail after fallbacks were correctly removed from the stage.
             *
             * The implementation materializes one reusable thread-local SwiGLU
             * tile and then dispatches the down projection by weight format.
             * M=1 uses the skinny kernel used by serial decode. FP32 M>1 uses
             * oneDNN's matrix path, while FP16/BF16 retain the mixed-input
             * skinny kernels because their activation rows remain FP32.
             */
            bool multiply_tensor_with_fused_swiglu(
                const TensorBase *gate,
                const TensorBase *up,
                TensorBase *output,
                int m, int n, int k,
                float alpha = 1.0f, float beta = 0.0f,
                DeviceWorkspaceManager *workspace = nullptr) override
            {
                (void)workspace;
                if (!weight_tensor_ || !gate || !up || !output ||
                    m <= 0 || n <= 0 || k <= 0)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] fused SwiGLU rejected invalid tensors or shape"
                              << " weight=" << (weight_tensor_ != nullptr)
                              << " gate=" << (gate != nullptr)
                              << " up=" << (up != nullptr)
                              << " output=" << (output != nullptr)
                              << " m=" << m << " n=" << n << " k=" << k);
                    return false;
                }
                if (gate->native_type() != TensorType::FP32 ||
                    up->native_type() != TensorType::FP32 ||
                    output->native_type() != TensorType::FP32)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] fused SwiGLU requires FP32 gate, up, and output tensors");
                    return false;
                }

                const size_t input_elements =
                    static_cast<size_t>(m) * static_cast<size_t>(k);
                const size_t output_elements =
                    static_cast<size_t>(m) * static_cast<size_t>(n);
                if (gate->numel() < input_elements ||
                    up->numel() < input_elements ||
                    output->numel() < output_elements)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] fused SwiGLU tensor capacity mismatch");
                    return false;
                }

                const float *gate_data = gate->data();
                const float *up_data = up->data();
                float *out_data = output->mutable_data();
                if (!gate_data || !up_data || !out_data)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] fused SwiGLU requires host-visible tensors");
                    return false;
                }

                thread_local std::vector<float> swiglu_scratch_tls;
                if (swiglu_scratch_tls.size() < input_elements)
                    swiglu_scratch_tls.resize(input_elements);
                if (numerical_policy_ == NumericalPolicy::GPUAlignedExpert)
                {
                    /*
                     * Materialize each SwiGLU word once through the exact
                     * CPU/CUDA/ROCm polynomial. GPU output blocks recompute
                     * this pure value independently, so reuse changes cost but
                     * not the published arithmetic program.
                     */
                    for (std::size_t index = 0; index < input_elements; ++index)
                    {
                        swiglu_scratch_tls[index] =
                            floating_expert_numerical_contract::swigluValue(
                                gate_data[index], up_data[index]);
                    }
                    return runGPUAlignedExpertDownProjection(
                        swiglu_scratch_tls.data(),
                        out_data,
                        m,
                        n,
                        k,
                        alpha,
                        beta);
                }
                if (m == 1)
                {
                    primitives::compute_swiglu_serial(
                        gate_data,
                        up_data,
                        swiglu_scratch_tls.data(),
                        static_cast<int>(input_elements));
                }
                else
                {
                    primitives::compute_swiglu(
                        gate_data,
                        up_data,
                        swiglu_scratch_tls.data(),
                        static_cast<int>(input_elements));
                }

                switch (weight_type_)
                {
                case TensorType::FP32:
                    if (m == 1)
                    {
                        return run_fp32_skinny_matmul(
                            swiglu_scratch_tls.data(),
                            weight_tensor_->data(),
                            out_data,
                            m,
                            n,
                            k,
                            /*transpose_B=*/true,
                            alpha,
                            beta,
                            nullptr);
                    }
                    return run_onednn_fp32_matmul(
                        swiglu_scratch_tls.data(),
                        weight_tensor_->data(),
                        out_data,
                        m,
                        n,
                        k,
                        /*transpose_B=*/true,
                        alpha,
                        beta,
                        nullptr);

                case TensorType::FP16:
                {
                    const auto *weights =
                        dynamic_cast<const FP16Tensor *>(weight_tensor_);
                    return weights && run_fp32xfp16_skinny_matmul(
                                          swiglu_scratch_tls.data(),
                                          weights->typed_data(),
                                          out_data,
                                          m,
                                          n,
                                          k,
                                          /*transpose_B=*/true,
                                          alpha,
                                          beta);
                }

                case TensorType::BF16:
                {
                    const auto *weights =
                        dynamic_cast<const BF16Tensor *>(weight_tensor_);
                    return weights && run_fp32xbf16_skinny_matmul(
                                          swiglu_scratch_tls.data(),
                                          weights->typed_data(),
                                          out_data,
                                          m,
                                          n,
                                          k,
                                          /*transpose_B=*/true,
                                          alpha,
                                          beta);
                }

                default:
                    LOG_ERROR("[FloatingPointGemmKernel] fused SwiGLU unsupported weight type "
                              << static_cast<int>(weight_type_));
                    return false;
                }
            }

            /**
             * @brief Grouped verifier SwiGLU + FP32 down projection.
             *
             * The verifier graph may carry an arbitrary runtime-M candidate batch, but state
             * publication must remain equivalent to serial decode.  This path
             * computes all SwiGLU rows with the shared CPU primitive, then uses
             * run_fp32_skinny_matmul(): work is parallelized over rows/columns
             * while each dot product walks K in the same order as one-token
             * decode.  It is therefore grouped, but not a hidden row replay.
             */
            bool multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                const TensorBase *gate,
                const TensorBase *up,
                TensorBase *output,
                int m, int n, int k,
                float alpha = 1.0f, float beta = 0.0f,
                DeviceWorkspaceManager *workspace = nullptr) override
            {
                (void)workspace;
                if (!weight_tensor_ || !gate || !up || !output ||
                    m < 1 || n <= 0 || k <= 0)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] grouped verifier SwiGLU rejected: weight="
                              << (weight_tensor_ != nullptr)
                              << " gate=" << (gate != nullptr)
                              << " up=" << (up != nullptr)
                              << " output=" << (output != nullptr)
                              << " m=" << m << " n=" << n << " k=" << k);
                    return false;
                }
                if (alpha != 1.0f || beta != 0.0f)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] grouped verifier SwiGLU only supports alpha=1,beta=0; got alpha="
                              << alpha << " beta=" << beta);
                    return false;
                }
                if (gate->native_type() != TensorType::FP32 ||
                    up->native_type() != TensorType::FP32)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] grouped verifier SwiGLU requires FP32 gate/up tensors");
                    return false;
                }
                if (output->native_type() != TensorType::FP32)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] grouped verifier SwiGLU output must be FP32");
                    return false;
                }

                const float *gate_data = gate->data();
                const float *up_data = up->data();
                float *out_data = output->mutable_data();
                if (!gate_data || !up_data || !out_data)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] grouped verifier SwiGLU requires host-visible FP32 tensors");
                    return false;
                }

                thread_local std::vector<float> swiglu_scratch_tls;
                const size_t elements =
                    static_cast<size_t>(m) * static_cast<size_t>(k);
                if (swiglu_scratch_tls.size() < elements)
                    swiglu_scratch_tls.resize(elements);
                const bool perf_enabled =
                    PerfStatsCollector::isDomainEnabled("kernel");
                auto perf_start = perf_enabled ? PerfStatsCollector::Clock::now()
                                               : PerfStatsCollector::Clock::time_point{};
                if (numerical_policy_ == NumericalPolicy::GPUAlignedExpert)
                {
                    for (std::size_t index = 0; index < elements; ++index)
                    {
                        swiglu_scratch_tls[index] =
                            floating_expert_numerical_contract::swigluValue(
                                gate_data[index], up_data[index]);
                    }
                }
                else
                {
                    primitives::compute_swiglu(
                        gate_data,
                        up_data,
                        swiglu_scratch_tls.data(),
                        static_cast<int>(elements));
                }
                recordVerifierTiming(
                    "cpu_fp32_verifier_swiglu_compute",
                    perf_start,
                    m,
                    n,
                    k,
                    1);

                perf_start = perf_enabled ? PerfStatsCollector::Clock::now()
                                          : PerfStatsCollector::Clock::time_point{};
                bool down_ok = false;
                const char *dtype_tag = "fp32";
                if (numerical_policy_ == NumericalPolicy::GPUAlignedExpert)
                {
                    down_ok = runGPUAlignedExpertDownProjection(
                        swiglu_scratch_tls.data(),
                        out_data,
                        m,
                        n,
                        k,
                        alpha,
                        beta);
                    if (weight_type_ == TensorType::FP16)
                        dtype_tag = "fp16";
                    else if (weight_type_ == TensorType::BF16)
                        dtype_tag = "bf16";
                }
                else if (weight_type_ == TensorType::FP32)
                {
                    down_ok = run_fp32_skinny_matmul(
                        swiglu_scratch_tls.data(),
                        weight_tensor_->data(),
                        out_data,
                        m,
                        n,
                        k,
                        /*transpose_B=*/true,
                        alpha,
                        beta,
                        nullptr);
                }
                else if (weight_type_ == TensorType::FP16)
                {
                    const auto *weights_fp16 = dynamic_cast<const FP16Tensor *>(weight_tensor_);
                    dtype_tag = "fp16";
                    down_ok = weights_fp16 && run_fp32xfp16_skinny_matmul(
                                                 swiglu_scratch_tls.data(),
                                                 weights_fp16->typed_data(),
                                                 out_data,
                                                 m,
                                                 n,
                                                 k,
                                                 /*transpose_B=*/true,
                                                 alpha,
                                                 beta);
                }
                else if (weight_type_ == TensorType::BF16)
                {
                    const auto *weights_bf16 = dynamic_cast<const BF16Tensor *>(weight_tensor_);
                    dtype_tag = "bf16";
                    down_ok = weights_bf16 && run_fp32xbf16_skinny_matmul(
                                                 swiglu_scratch_tls.data(),
                                                 weights_bf16->typed_data(),
                                                 out_data,
                                                 m,
                                                 n,
                                                 k,
                                                 /*transpose_B=*/true,
                                                 alpha,
                                                 beta);
                }

                if (!down_ok)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] grouped verifier SwiGLU down projection failed dtype="
                              << dtype_tag);
                    return false;
                }
                recordVerifierTiming(
                    "cpu_fp32_verifier_swiglu_down_matmul",
                    perf_start,
                    m,
                    n,
                    k,
                    1);

                if (PerfStatsCollector::isDomainEnabled("kernel"))
                {
                    PerfStatsCollector::addCounter(
                        "kernel",
                        "cpu_floating_grouped_verifier_swiglu_down_calls",
                        1.0,
                        "gemm",
                        "cpu",
                        PerfStatsCollector::Tags{
                            {"dtype", dtype_tag},
                            {"m", std::to_string(m)},
                            {"n", std::to_string(n)},
                            {"k", std::to_string(k)}});
                    if (weight_type_ == TensorType::FP32)
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "cpu_fp32_grouped_verifier_swiglu_down_calls",
                            1.0,
                            "gemm",
                            "cpu",
                            PerfStatsCollector::Tags{
                                {"m", std::to_string(m)},
                                {"n", std::to_string(n)},
                                {"k", std::to_string(k)}});
                    }
                }
                return true;
            }

            /**
             * @brief Report the first-class CPU floating projection-bundle path.
             *
             * Floating projections do not share activation quantization, but the
             * bundle still owns validation and execution as one typed operation.
             * This prevents stages from rebuilding a failed bundle through the
             * polymorphic single-projection entry point.
             */
            bool supports_fused_projection() const override { return true; }

            /**
             * @brief Execute a homogeneous floating-point projection bundle.
             *
             * Every descriptor is authenticated before the first output is
             * touched. Backend-native engines invoke the format-specific CPU
             * primitive directly. GPU-aligned expert engines instead execute the
             * shared 256-lane CPU/CUDA/ROCm numerical contract; bundle fusion
             * must not erase the placement-invariant policy carried by each
             * prepared engine.
             *
             * @param input Shared activation matrix with shape at least `[m,k]`.
             * @param projections Homogeneous FP32, FP16, or BF16 projections.
             * @param m Runtime row count.
             * @param k Shared reduction width.
             * @param mpi_ctx Unused for this rank-local CPU implementation.
             * @param workspace Unused because oneDNN owns its internal scratch.
             * @return true only after every projection completed successfully.
             */
            bool multiply_fused_tensor(
                const TensorBase *input,
                const std::vector<TensorProjectionDesc> &projections,
                int m,
                int k,
                const IMPIContext *mpi_ctx = nullptr,
                DeviceWorkspaceManager *workspace = nullptr) override
            {
                (void)mpi_ctx;
                (void)workspace;

                const bool mixed_fp32_activation =
                    input && input->native_type() == TensorType::FP32 &&
                    (weight_type_ == TensorType::FP16 ||
                     weight_type_ == TensorType::BF16);
                if (!weight_tensor_ || !input || projections.empty() ||
                    m <= 0 || k <= 0 ||
                    (input->native_type() != weight_type_ &&
                     !mixed_fp32_activation) ||
                    (numerical_policy_ == NumericalPolicy::GPUAlignedExpert &&
                     input->native_type() != TensorType::FP32) ||
                    input->numel() < static_cast<size_t>(m) * k)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] Invalid fused projection bundle"
                              << " activation="
                              << (input
                                      ? static_cast<int>(input->native_type())
                                      : -1)
                              << " weight="
                              << static_cast<int>(weight_type_)
                              << " m=" << m << " k=" << k
                              << " projections=" << projections.size());
                    return false;
                }

                // Validate the complete transaction before any projection writes.
                for (size_t index = 0; index < projections.size(); ++index)
                {
                    const auto &projection = projections[index];
                    const auto *kernel =
                        dynamic_cast<const FloatingPointGemmKernel *>(
                            projection.kernel);
                    if (!kernel || !kernel->weight_tensor_ ||
                        kernel->weight_type_ != weight_type_ ||
                        kernel->numerical_policy_ != numerical_policy_ ||
                        !projection.output || projection.n <= 0 ||
                        projection.output->native_type() != TensorType::FP32 ||
                        projection.output->numel() <
                            static_cast<size_t>(m) * projection.n ||
                        (projection.bias &&
                         (weight_type_ != TensorType::FP32 ||
                          projection.bias->native_type() != TensorType::FP32 ||
                          projection.bias->numel() <
                              static_cast<size_t>(projection.n))))
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Fused projection contract "
                                  "mismatch at descriptor " << index);
                        return false;
                    }
                }

                for (const auto &projection : projections)
                {
                    auto *kernel = static_cast<FloatingPointGemmKernel *>(
                        projection.kernel);
                    float *output = projection.output->mutable_data();
                    if (!output)
                        return false;

                    bool succeeded = false;
                    switch (weight_type_)
                    {
                    case TensorType::FP32:
                        succeeded =
                            numerical_policy_ == NumericalPolicy::GPUAlignedExpert
                                ? run_gpu_aligned_expert_fp32_matmul<
                                      TensorType::FP32>(
                                      input->data(),
                                      kernel->weight_tensor_->data(),
                                      output,
                                      m,
                                      projection.n,
                                      k,
                                      /*alpha=*/1.0f,
                                      /*beta=*/0.0f,
                                      projection.bias
                                          ? projection.bias->data()
                                          : nullptr)
                                : run_onednn_fp32_matmul(
                                      input->data(),
                                      kernel->weight_tensor_->data(),
                                      output,
                                      m,
                                      projection.n,
                                      k,
                                      /*transpose_B=*/true,
                                      /*alpha=*/1.0f,
                                      /*beta=*/0.0f,
                                      projection.bias
                                          ? projection.bias->data()
                                          : nullptr);
                        break;
                    case TensorType::FP16:
                    {
                        const auto *typed_weight =
                            dynamic_cast<const FP16Tensor *>(
                                kernel->weight_tensor_);
                        if (mixed_fp32_activation)
                        {
                            succeeded = typed_weight &&
                                (numerical_policy_ ==
                                         NumericalPolicy::GPUAlignedExpert
                                     ? run_gpu_aligned_expert_fp32_matmul<
                                           TensorType::FP16>(
                                           input->data(),
                                           typed_weight->typed_data(),
                                           output,
                                           m,
                                           projection.n,
                                           k)
                                     : run_fp32xfp16_skinny_matmul(
                                           input->data(),
                                           typed_weight->typed_data(),
                                           output,
                                           m,
                                           projection.n,
                                           k,
                                           /*transpose_B=*/true,
                                           /*alpha=*/1.0f,
                                           /*beta=*/0.0f));
                        }
                        else
                        {
                            const auto *typed_input =
                                dynamic_cast<const FP16Tensor *>(input);
                            succeeded = typed_input && typed_weight &&
                                run_onednn_fp16_matmul(
                                    typed_input->typed_data(),
                                    typed_weight->typed_data(),
                                    output,
                                    m,
                                    projection.n,
                                    k,
                                    /*transpose_B=*/true,
                                    /*alpha=*/1.0f,
                                    /*beta=*/0.0f);
                        }
                        break;
                    }
                    case TensorType::BF16:
                    {
                        const auto *typed_weight =
                            dynamic_cast<const BF16Tensor *>(
                                kernel->weight_tensor_);
                        if (mixed_fp32_activation)
                        {
                            succeeded = typed_weight &&
                                (numerical_policy_ ==
                                         NumericalPolicy::GPUAlignedExpert
                                     ? run_gpu_aligned_expert_fp32_matmul<
                                           TensorType::BF16>(
                                           input->data(),
                                           typed_weight->typed_data(),
                                           output,
                                           m,
                                           projection.n,
                                           k)
                                     : run_fp32xbf16_skinny_matmul(
                                           input->data(),
                                           typed_weight->typed_data(),
                                           output,
                                           m,
                                           projection.n,
                                           k,
                                           /*transpose_B=*/true,
                                           /*alpha=*/1.0f,
                                           /*beta=*/0.0f));
                        }
                        else
                        {
                            const auto *typed_input =
                                dynamic_cast<const BF16Tensor *>(input);
                            succeeded = typed_input && typed_weight &&
                                run_onednn_bf16_matmul(
                                    typed_input->typed_data(),
                                    typed_weight->typed_data(),
                                    output,
                                    m,
                                    projection.n,
                                    k,
                                    /*transpose_B=*/true,
                                    /*alpha=*/1.0f,
                                    /*beta=*/0.0f);
                        }
                        break;
                    }
                    default:
                        return false;
                    }

                    if (!succeeded)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Fused projection "
                                  "execution failed for "
                                  << (projection.name ? projection.name
                                                      : "unnamed"));
                        return false;
                    }
                }

                if (PerfStatsCollector::isDomainEnabled("kernel"))
                {
                    PerfStatsCollector::addCounter(
                        "kernel",
                        "cpu_floating_fused_projection_calls",
                        1.0,
                        "gemm",
                        "cpu",
                        {{"m", std::to_string(m)},
                         {"k", std::to_string(k)},
                         {"projections", std::to_string(projections.size())}});
                }
                return true;
            }

            /**
             * @brief Group verifier rows while preserving serial decode dot-product order.
             *
             * Phase 9.8 verifier graphs evaluate runtime-M candidate rows together,
             * but publication-capable recurrent state must match the row-by-row
             * decode contract.  FP32, FP16, and BF16 weights use skinny
             * primitives that parallelize over rows and output columns in one
             * OpenMP region, while every dot product still walks K in the same
             * order as the M=1 decode GEMV.  This is intentionally not a hidden
             * serial row loop.
             */
            bool multiply_fused_verifier_rows_decode_equivalent(
                const TensorBase *input,
                const std::vector<TensorProjectionDesc> &projections,
                int m, int k,
                const IMPIContext *mpi_ctx = nullptr,
                DeviceWorkspaceManager *workspace = nullptr) override
            {
                (void)mpi_ctx;
                (void)workspace;

                if (!weight_tensor_ || !input || m <= 1 || k <= 0 || projections.empty())
                {
                    LOG_ERROR("[FloatingPointGemmKernel] grouped verifier projection rejected: weight="
                              << (weight_tensor_ != nullptr) << " input=" << (input != nullptr)
                              << " m=" << m << " k=" << k
                              << " projections=" << projections.size());
                    return false;
                }

                const bool mixed_fp32_activation =
                    input->native_type() == TensorType::FP32 &&
                    (weight_type_ == TensorType::FP16 ||
                     weight_type_ == TensorType::BF16);
                if (input->native_type() != weight_type_ &&
                    !mixed_fp32_activation)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] grouped verifier projection rejected: unsupported activation type "
                              << static_cast<int>(input->native_type())
                              << " for weight type "
                              << static_cast<int>(weight_type_));
                    return false;
                }
                if (numerical_policy_ == NumericalPolicy::GPUAlignedExpert &&
                    input->native_type() != TensorType::FP32)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] grouped GPU-aligned expert projection requires FP32 transported rows");
                    return false;
                }

                const float *input_fp32 = nullptr;
                const uint16_t *input_fp16 = nullptr;
                const uint16_t *input_bf16 = nullptr;
                const char *dtype_tag = "unknown";
                const char *timing_name = "cpu_floating_verifier_projection_matmul";

                switch (weight_type_)
                {
                case TensorType::FP32:
                    input_fp32 = input->data();
                    dtype_tag = "fp32";
                    timing_name = "cpu_fp32_verifier_projection_matmul";
                    if (!input_fp32)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] grouped verifier projection rejected: input has no host FP32 data");
                        return false;
                    }
                    break;
                case TensorType::FP16:
                {
                    if (mixed_fp32_activation)
                    {
                        input_fp32 = input->data();
                        dtype_tag = "fp32xfp16";
                        timing_name =
                            "cpu_fp32xfp16_verifier_projection_matmul";
                        if (!input_fp32)
                        {
                            LOG_ERROR("[FloatingPointGemmKernel] grouped verifier projection rejected: input has no host FP32 data");
                            return false;
                        }
                    }
                    else
                    {
                        const auto *typed =
                            dynamic_cast<const FP16Tensor *>(input);
                        input_fp16 = typed ? typed->typed_data() : nullptr;
                        dtype_tag = "fp16";
                        timing_name =
                            "cpu_fp16_verifier_projection_matmul";
                        if (!input_fp16)
                        {
                            LOG_ERROR("[FloatingPointGemmKernel] grouped verifier projection rejected: input has no host FP16 data");
                            return false;
                        }
                    }
                    break;
                }
                case TensorType::BF16:
                {
                    if (mixed_fp32_activation)
                    {
                        input_fp32 = input->data();
                        dtype_tag = "fp32xbf16";
                        timing_name =
                            "cpu_fp32xbf16_verifier_projection_matmul";
                        if (!input_fp32)
                        {
                            LOG_ERROR("[FloatingPointGemmKernel] grouped verifier projection rejected: input has no host FP32 data");
                            return false;
                        }
                    }
                    else
                    {
                        const auto *typed =
                            dynamic_cast<const BF16Tensor *>(input);
                        input_bf16 = typed ? typed->typed_data() : nullptr;
                        dtype_tag = "bf16";
                        timing_name =
                            "cpu_bf16_verifier_projection_matmul";
                        if (!input_bf16)
                        {
                            LOG_ERROR("[FloatingPointGemmKernel] grouped verifier projection rejected: input has no host BF16 data");
                            return false;
                        }
                    }
                    break;
                }
                default:
                    LOG_ERROR("[FloatingPointGemmKernel] grouped verifier projection unsupported weight type "
                              << static_cast<int>(weight_type_));
                    return false;
                }

                for (size_t i = 0; i < projections.size(); ++i)
                {
                    const auto &proj = projections[i];
                    auto *fp = dynamic_cast<FloatingPointGemmKernel *>(proj.kernel);
                    if (!fp || !fp->weight_tensor_ || fp->weight_type_ != weight_type_ ||
                        fp->numerical_policy_ != numerical_policy_ ||
                        !proj.output || proj.n <= 0)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] grouped verifier projection rejected at projection "
                                  << i << ": fp=" << (fp != nullptr)
                                  << " weight=" << (fp && fp->weight_tensor_)
                                  << " same_type=" << (fp && fp->weight_type_ == weight_type_)
                                  << " output=" << (proj.output != nullptr)
                                  << " n=" << proj.n);
                        return false;
                    }
                    if (proj.output->native_type() != TensorType::FP32)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] grouped verifier projection rejected at projection "
                                  << i << ": output must be FP32, got "
                                  << static_cast<int>(proj.output->native_type()));
                        return false;
                    }

                    float *out_data = proj.output->mutable_data();
                    if (!out_data)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] grouped verifier projection rejected at projection "
                                  << i << ": output has no host FP32 data");
                        return false;
                    }

                    const float *bias_ptr = proj.bias ? proj.bias->data() : nullptr;
                    if (proj.bias && !bias_ptr)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] grouped verifier projection rejected at projection "
                                  << i << ": bias has no host FP32 data");
                        return false;
                    }
                    if (proj.bias && weight_type_ != TensorType::FP32)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] grouped verifier FP16/BF16 projection does not support bias yet");
                        return false;
                    }

                    const bool perf_enabled =
                        PerfStatsCollector::isDomainEnabled("kernel");
                    const auto perf_start = perf_enabled ? PerfStatsCollector::Clock::now()
                                                         : PerfStatsCollector::Clock::time_point{};
                    bool projection_ok = false;
                    if (weight_type_ == TensorType::FP32)
                    {
                        projection_ok =
                            numerical_policy_ == NumericalPolicy::GPUAlignedExpert
                                ? run_gpu_aligned_expert_fp32_matmul<
                                      TensorType::FP32>(
                                      input_fp32,
                                      fp->weight_tensor_->data(),
                                      out_data,
                                      m,
                                      proj.n,
                                      k,
                                      /*alpha=*/1.0f,
                                      /*beta=*/0.0f,
                                      bias_ptr)
                                : run_fp32_skinny_matmul(
                                      input_fp32,
                                      fp->weight_tensor_->data(),
                                      out_data,
                                      m,
                                      proj.n,
                                      k,
                                      /*transpose_B=*/true,
                                      /*alpha=*/1.0f,
                                      /*beta=*/0.0f,
                                      bias_ptr);
                    }
                    else if (weight_type_ == TensorType::FP16)
                    {
                        const auto *weight_fp16 = dynamic_cast<const FP16Tensor *>(fp->weight_tensor_);
                        projection_ok = weight_fp16 &&
                            (mixed_fp32_activation
                                 ? (numerical_policy_ ==
                                            NumericalPolicy::GPUAlignedExpert
                                        ? run_gpu_aligned_expert_fp32_matmul<
                                              TensorType::FP16>(
                                              input_fp32,
                                              weight_fp16->typed_data(),
                                              out_data,
                                              m,
                                              proj.n,
                                              k)
                                        : run_fp32xfp16_skinny_matmul(
                                              input_fp32,
                                              weight_fp16->typed_data(),
                                              out_data,
                                              m,
                                              proj.n,
                                              k,
                                              /*transpose_B=*/true))
                                 : run_fp16_skinny_matmul(
                                       input_fp16,
                                       weight_fp16->typed_data(),
                                       out_data,
                                       m,
                                       proj.n,
                                       k,
                                       /*transpose_B=*/true));
                    }
                    else if (weight_type_ == TensorType::BF16)
                    {
                        const auto *weight_bf16 = dynamic_cast<const BF16Tensor *>(fp->weight_tensor_);
                        projection_ok = weight_bf16 &&
                            (mixed_fp32_activation
                                 ? (numerical_policy_ ==
                                            NumericalPolicy::GPUAlignedExpert
                                        ? run_gpu_aligned_expert_fp32_matmul<
                                              TensorType::BF16>(
                                              input_fp32,
                                              weight_bf16->typed_data(),
                                              out_data,
                                              m,
                                              proj.n,
                                              k)
                                        : run_fp32xbf16_skinny_matmul(
                                              input_fp32,
                                              weight_bf16->typed_data(),
                                              out_data,
                                              m,
                                              proj.n,
                                              k,
                                              /*transpose_B=*/true))
                                 : run_bf16_skinny_matmul(
                                       input_bf16,
                                       weight_bf16->typed_data(),
                                       out_data,
                                       m,
                                       proj.n,
                                       k,
                                       /*transpose_B=*/true));
                    }

                    if (!projection_ok)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] grouped verifier projection failed at projection "
                                  << i << " n=" << proj.n
                                  << " dtype=" << dtype_tag);
                        return false;
                    }
                    recordVerifierTiming(
                        timing_name,
                        perf_start,
                        m,
                        proj.n,
                        k,
                        static_cast<int>(projections.size()));
                }

                if (PerfStatsCollector::isDomainEnabled("kernel"))
                {
                    PerfStatsCollector::addCounter(
                        "kernel",
                        "cpu_floating_grouped_verifier_projection_calls",
                        1.0,
                        "gemm",
                        "cpu",
                        PerfStatsCollector::Tags{
                            {"dtype", dtype_tag},
                            {"m", std::to_string(m)},
                            {"k", std::to_string(k)},
                            {"projections", std::to_string(projections.size())}});

                    if (weight_type_ != TensorType::FP32)
                    {
                        return true;
                    }

                    PerfStatsCollector::addCounter(
                        "kernel",
                        "cpu_fp32_grouped_verifier_projection_calls",
                        1.0,
                        "gemm",
                        "cpu",
                        PerfStatsCollector::Tags{
                            {"m", std::to_string(m)},
                            {"k", std::to_string(k)},
                            {"projections", std::to_string(projections.size())}});
                }
                return true;
            }

            /**
             * @brief Record coarse CPU FP32 verifier timing samples.
             *
             * The timers sit around grouped verifier projection families rather
             * than inside every dot product.  They are enabled only when the
             * structured perf collector is active, and they preserve enough shape
             * tags to compare FP32 GDN projection cost with NativeVNNI cost.
             */
            static void recordVerifierTiming(
                const char *name,
                PerfStatsCollector::Clock::time_point start,
                int m,
                int n,
                int k,
                int projections)
            {
                if (!PerfStatsCollector::isDomainEnabled("kernel"))
                    return;

                const auto end = PerfStatsCollector::Clock::now();
                const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                PerfStatsCollector::Tags tags{
                    {"m", std::to_string(m)},
                    {"k", std::to_string(k)},
                    {"projections", std::to_string(projections)}};
                if (n > 0)
                    tags.emplace("n", std::to_string(n));
                PerfStatsCollector::recordTimingNs(
                    "kernel",
                    name ? name : "cpu_fp32_verifier_unknown",
                    ns > 0 ? static_cast<uint64_t>(ns) : 0,
                    "gemm",
                    "cpu",
                    std::move(tags));
            }

            // ========== Typed Activation Interface ==========

            /**
             * @brief Typed activation GEMM (FP16×FP16 or BF16×BF16)
             */
            bool multiply_activations_typed_impl(
                const void *A, const void *B, float *C,
                int m, int n, int k,
                bool transpose_B,
                float alpha, float beta,
                const IMPIContext *mpi_ctx,
                int device_idx,
                ActivationFormat format_A, ActivationFormat format_B)
            {
                (void)mpi_ctx;
                (void)device_idx;

                // FP16 × FP16
                if (format_A == ActivationFormat::FP16 && format_B == ActivationFormat::FP16)
                {
                    const uint16_t *A_ptr = static_cast<const uint16_t *>(A);
                    const uint16_t *B_ptr = static_cast<const uint16_t *>(B);

                    if (!B_ptr && weight_tensor_)
                    {
                        if (weight_type_ != TensorType::FP16)
                        {
                            LOG_ERROR("[FloatingPointGemmKernel] FP16 activation requires FP16 weights");
                            return false;
                        }
                        const auto *fp16_tensor = dynamic_cast<const FP16Tensor *>(weight_tensor_);
                        B_ptr = fp16_tensor ? fp16_tensor->typed_data() : nullptr;
                    }

                    if (!B_ptr)
                        return false;
                    return run_onednn_fp16_matmul(A_ptr, B_ptr, C, m, n, k, transpose_B, alpha, beta);
                }

                // BF16 × BF16
                if (format_A == ActivationFormat::BF16 && format_B == ActivationFormat::BF16)
                {
                    const uint16_t *A_ptr = static_cast<const uint16_t *>(A);
                    const uint16_t *B_ptr = static_cast<const uint16_t *>(B);

                    if (!B_ptr && weight_tensor_)
                    {
                        if (weight_type_ != TensorType::BF16)
                        {
                            LOG_ERROR("[FloatingPointGemmKernel] BF16 activation requires BF16 weights");
                            return false;
                        }
                        const auto *bf16_tensor = dynamic_cast<const BF16Tensor *>(weight_tensor_);
                        B_ptr = bf16_tensor ? bf16_tensor->typed_data() : nullptr;
                    }

                    if (!B_ptr)
                        return false;
                    return run_onednn_bf16_matmul(A_ptr, B_ptr, C, m, n, k, transpose_B, alpha, beta);
                }

                LOG_ERROR("[FloatingPointGemmKernel] Unsupported activation format combination");
                return false;
            }

            bool multiply_with_softmax_typed_impl(
                const void *A, const void *B, float *C,
                int m, int n, int k,
                float scale,
                bool transpose_B,
                int softmax_axis,
                const float *mask,
                bool is_causal,
                const IMPIContext *mpi_ctx,
                int device_idx,
                ActivationFormat format_A, ActivationFormat format_B)
            {
                // 1. GEMM with scale
                if (!multiply_activations_typed_impl(A, B, C, m, n, k, transpose_B, scale, 0.0f, mpi_ctx, device_idx, format_A, format_B))
                {
                    return false;
                }

                // 2. Add mask
                if (mask)
                {
                    auto add_mask_work = [&]()
                    {
#pragma omp for collapse(2) schedule(static)
                        for (int i = 0; i < m; ++i)
                        {
                            for (int j = 0; j < n; ++j)
                            {
                                C[i * n + j] += mask[i * n + j];
                            }
                        }
                    };
                    OMP_WORKSHARE_REGION(add_mask_work);
                }

                // 3. Causal mask
                if (is_causal)
                {
                    auto causal_mask_work = [&]()
                    {
#pragma omp for schedule(static)
                        for (int i = 0; i < m; ++i)
                        {
                            for (int j = i + 1; j < n; ++j)
                            {
                                C[i * n + j] = -std::numeric_limits<float>::infinity();
                            }
                        }
                    };
                    OMP_WORKSHARE_REGION(causal_mask_work);
                }

                // 4. Softmax
                // Use our robust primitives that handle -inf correctly
                primitives::softmax_row_major_fp32(C, m, n, false, 1.0f, true);

                return true;
            }

            bool multiply_activations(
                const float *A, const float *B, float *C,
                int m, int n, int k,
                bool transpose_B, float alpha, float beta,
                const IMPIContext *mpi_ctx, int device_idx)
            {
                if (!B && weight_tensor_)
                {
                    return run_onednn_fp32_matmul(A, nullptr, C, m, n, k, transpose_B, alpha, beta);
                }
                return run_onednn_fp32_matmul(A, B, C, m, n, k, transpose_B, alpha, beta);
            }

            /**
             * @brief Tensor-based activation×activation GEMM with type-aware dispatch
             *
             * Handles all input tensor type combinations:
             * - FP32 × FP32: Direct FP32 matmul
             * - BF16 × BF16: OneDNN bf16bf16f32
             * - FP16 × FP16: OneDNN fp16fp16f32
             * - Q8_1 × Q8_1: Dequant to FP32, then FP32 matmul (attention scores always FP32)
             *
             * Output is always written to C via mutable_data() (FP32 for attention scores).
             */
            bool multiply_activations_tensor(
                const TensorBase *A, const TensorBase *B, TensorBase *C,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                const auto &a_shape = A->shape();
                const auto &b_shape = B->shape();
                int m = static_cast<int>(a_shape[0]);
                int k = static_cast<int>(a_shape.size() > 1 ? a_shape[1] : 1);
                int n = transpose_B ? static_cast<int>(b_shape[0]) : static_cast<int>(b_shape.size() > 1 ? b_shape[1] : 1);

                const TensorType a_type = A->native_type();
                const TensorType b_type = B->native_type();

                // FP32 path (most common for attention)
                if (a_type == TensorType::FP32 && b_type == TensorType::FP32)
                {
                    return run_onednn_fp32_matmul(
                        A->data(), B->data(), C->mutable_data(),
                        m, n, k, transpose_B, alpha, beta);
                }

                // BF16 path
                if (a_type == TensorType::BF16 && b_type == TensorType::BF16)
                {
                    const auto *A_bf16 = dynamic_cast<const BF16Tensor *>(A);
                    const auto *B_bf16 = dynamic_cast<const BF16Tensor *>(B);
                    if (A_bf16 && B_bf16)
                    {
                        return run_onednn_bf16_matmul(
                            A_bf16->typed_data(), B_bf16->typed_data(), C->mutable_data(),
                            m, n, k, transpose_B, alpha, beta);
                    }
                }

                // FP16 path
                if (a_type == TensorType::FP16 && b_type == TensorType::FP16)
                {
                    const auto *A_fp16 = dynamic_cast<const FP16Tensor *>(A);
                    const auto *B_fp16 = dynamic_cast<const FP16Tensor *>(B);
                    if (A_fp16 && B_fp16)
                    {
                        return run_onednn_fp16_matmul(
                            A_fp16->typed_data(), B_fp16->typed_data(), C->mutable_data(),
                            m, n, k, transpose_B, alpha, beta);
                    }
                }

                // Q8_1 path: Use fp32_data() for explicit dequantization
                // Attention scores are always FP32 (softmax needs FP32 precision)
                if (a_type == TensorType::Q8_1 && b_type == TensorType::Q8_1)
                {
                    return run_onednn_fp32_matmul(
                        A->fp32_data(), B->fp32_data(), C->mutable_data(),
                        m, n, k, transpose_B, alpha, beta);
                }

                // Fallback: Convert to FP32 via data()
                return run_onednn_fp32_matmul(
                    A->data(), B->data(), C->mutable_data(),
                    m, n, k, transpose_B, alpha, beta);
            }

            bool multiply_with_softmax(
                const float *A, const float *B, float *C,
                int m, int n, int k,
                bool transpose_B, int softmax_axis, const float *mask,
                const IMPIContext *mpi_ctx, int device_idx)
            {
                // 1. GEMM: C = A * B
                // Direct weight GEMM via stored weight tensor
                if (!run_onednn_fp32_matmul(A, nullptr, C, m, n, k, transpose_B, 1.0f, 0.0f))
                {
                    return false;
                }

                // 2. Add mask if provided
                if (mask)
                {
                    auto add_mask_work = [&]()
                    {
#pragma omp for collapse(2) schedule(static)
                        for (int i = 0; i < m; ++i)
                        {
                            for (int j = 0; j < n; ++j)
                            {
                                C[i * n + j] += mask[i * n + j];
                            }
                        }
                    };
                    OMP_WORKSHARE_REGION(add_mask_work);
                }

                // 3. Softmax
                // Assuming contiguous C for now, or stride=n
                // Use our robust primitives that handle -inf correctly
                primitives::softmax_row_major_fp32(C, m, n, false, 1.0f, true);

                return true;
            }

            /**
             * @brief Fused strided GEMM + softmax for attention Q@K^T computation
             *
             * Computes: C = Softmax(scale * A @ B^T + mask)
             * Used by CpuAttentionKernelT for Q@K^T with causal masking
             */
            bool multiply_with_softmax_strided_typed_impl(
                const void *A, const void *B, float *C,
                int m, int n, int k,
                int lda, int ldb, int ldc,
                float scale,
                bool transpose_B,
                int softmax_axis,
                const float *mask,
                bool is_causal,
                const IMPIContext *mpi_ctx,
                int device_idx,
                ActivationFormat format_A,
                ActivationFormat format_B) override
            {
                (void)mpi_ctx;
                (void)device_idx;
                (void)softmax_axis; // We always do row-wise softmax

                // Step 1: Strided GEMM with scaling
                bool gemm_ok = false;

                // Handle homogeneous formats
                if (format_A == format_B)
                {
                    switch (format_A)
                    {
                    case ActivationFormat::FP32:
                        gemm_ok = run_onednn_fp32_matmul_strided(
                            static_cast<const float *>(A),
                            static_cast<const float *>(B),
                            C, m, n, k, lda, ldb, ldc, transpose_B, scale, 0.0f);
                        break;

                    case ActivationFormat::BF16:
                        gemm_ok = run_onednn_bf16_matmul_strided(
                            static_cast<const uint16_t *>(A),
                            static_cast<const uint16_t *>(B),
                            C, m, n, k, lda, ldb, ldc, transpose_B, scale, 0.0f);
                        break;

                    case ActivationFormat::FP16:
                        gemm_ok = run_onednn_fp16_matmul_strided(
                            static_cast<const uint16_t *>(A),
                            static_cast<const uint16_t *>(B),
                            C, m, n, k, lda, ldb, ldc, transpose_B, scale, 0.0f);
                        break;

                    default:
                        LOG_ERROR("[FloatingPointGemmKernel] Unsupported format for fused GEMM+softmax: "
                                  << static_cast<int>(format_A));
                        return false;
                    }
                }
                else
                {
                    LOG_ERROR("[FloatingPointGemmKernel] Fused GEMM+softmax requires matching formats: A="
                              << static_cast<int>(format_A) << " B=" << static_cast<int>(format_B));
                    return false;
                }

                if (!gemm_ok)
                {
                    return false;
                }

                // Step 2: Apply mask if provided (add to scores)
                // The mask has shape [m, n] with stride ldc
                if (mask)
                {
                    for (int row = 0; row < m; ++row)
                    {
                        for (int col = 0; col < n; ++col)
                        {
                            C[row * ldc + col] += mask[row * n + col];
                        }
                    }
                }

                // Step 3: Apply causal mask (set future positions to -inf)
                if (is_causal)
                {
                    for (int row = 0; row < m; ++row)
                    {
                        for (int col = row + 1; col < n; ++col)
                        {
                            C[row * ldc + col] = -std::numeric_limits<float>::infinity();
                        }
                    }
                }

                // Step 4: Apply row-wise softmax
                // Use our robust primitives that handle -inf correctly (unlike OneDNN)
                if (ldc == n)
                {
                    primitives::softmax_row_major_fp32(C, m, n, false, 1.0f, true);
                }
                else
                {
                    auto softmax_row_work = [&]()
                    {
#pragma omp for schedule(static)
                        for (int r = 0; r < m; ++r)
                        {
                            primitives::softmax_row_fp32(C + r * ldc, n, false, 1.0f, -1);
                        }
                    };
                    OMP_WORKSHARE_REGION(softmax_row_work);
                }

                return true;
            }

            bool multiply_activations_strided(
                const float *A, const float *B, float *C,
                int m, int n, int k,
                int lda, int ldb, int ldc,
                bool transpose_B, float alpha, float beta,
                const IMPIContext *mpi_ctx, int device_idx) override
            {
                (void)mpi_ctx;
                (void)device_idx;

                return run_onednn_fp32_matmul_strided(A, B, C, m, n, k, lda, ldb, ldc, transpose_B, alpha, beta);
            }

            bool multiply_activations_strided_typed_impl(
                const void *A, const void *B, float *C,
                int m, int n, int k,
                int lda, int ldb, int ldc,
                bool transpose_B, float alpha, float beta,
                const IMPIContext *mpi_ctx, int device_idx,
                ActivationFormat format_A, ActivationFormat format_B) override
            {
                (void)mpi_ctx;
                (void)device_idx;

                // Handle homogeneous formats (same type for A and B)
                if (format_A == format_B)
                {
                    switch (format_A)
                    {
                    case ActivationFormat::FP32:
                        return run_onednn_fp32_matmul_strided(
                            static_cast<const float *>(A),
                            static_cast<const float *>(B),
                            C, m, n, k, lda, ldb, ldc, transpose_B, alpha, beta);

                    case ActivationFormat::BF16:
                        return run_onednn_bf16_matmul_strided(
                            static_cast<const uint16_t *>(A),
                            static_cast<const uint16_t *>(B),
                            C, m, n, k, lda, ldb, ldc, transpose_B, alpha, beta);

                    case ActivationFormat::FP16:
                        return run_onednn_fp16_matmul_strided(
                            static_cast<const uint16_t *>(A),
                            static_cast<const uint16_t *>(B),
                            C, m, n, k, lda, ldb, ldc, transpose_B, alpha, beta);

                    default:
                        LOG_ERROR("[FloatingPointGemmKernel] Unsupported homogeneous format: "
                                  << static_cast<int>(format_A));
                        return false;
                    }
                }

                // Handle mixed-precision formats (FP32 scores × typed V for attention)
                if (format_A == ActivationFormat::FP32)
                {
                    switch (format_B)
                    {
                    case ActivationFormat::BF16:
                        return run_onednn_fp32_bf16_matmul_strided(
                            static_cast<const float *>(A),
                            static_cast<const uint16_t *>(B),
                            C, m, n, k, lda, ldb, ldc, transpose_B, alpha, beta);

                    case ActivationFormat::FP16:
                        return run_onednn_fp32_fp16_matmul_strided(
                            static_cast<const float *>(A),
                            static_cast<const uint16_t *>(B),
                            C, m, n, k, lda, ldb, ldc, transpose_B, alpha, beta);

                    default:
                        LOG_ERROR("[FloatingPointGemmKernel] Unsupported mixed format: FP32×"
                                  << static_cast<int>(format_B));
                        return false;
                    }
                }

                LOG_ERROR("[FloatingPointGemmKernel] Unsupported format combination: A="
                          << static_cast<int>(format_A) << " B=" << static_cast<int>(format_B));
                return false;
            }

        private:
            /**
             * @brief Down-project precomputed SwiGLU rows with the GPU tree.
             *
             * @param swiglu Row-major placement-invariant SwiGLU values `[M,K]`.
             * @param output Row-major FP32 destination `[M,N]`.
             * @param m Runtime row count.
             * @param n Output width.
             * @param k Reduction width.
             * @param alpha Output scale.
             * @param beta Existing-output scale.
             * @return True after the format-specific tree completed.
             */
            bool runGPUAlignedExpertDownProjection(
                const float *swiglu,
                float *output,
                int m,
                int n,
                int k,
                float alpha,
                float beta) const
            {
                switch (weight_type_)
                {
                case TensorType::FP32:
                {
                    const float *weights = weight_tensor_->data();
                    return weights &&
                           run_gpu_aligned_expert_fp32_matmul<
                               TensorType::FP32>(
                               swiglu,
                               weights,
                               output,
                               m,
                               n,
                               k,
                               alpha,
                               beta);
                }
                case TensorType::FP16:
                {
                    const auto *weights =
                        dynamic_cast<const FP16Tensor *>(weight_tensor_);
                    return weights &&
                           run_gpu_aligned_expert_fp32_matmul<
                               TensorType::FP16>(
                               swiglu,
                               weights->typed_data(),
                               output,
                               m,
                               n,
                               k,
                               alpha,
                               beta);
                }
                case TensorType::BF16:
                {
                    const auto *weights =
                        dynamic_cast<const BF16Tensor *>(weight_tensor_);
                    return weights &&
                           run_gpu_aligned_expert_fp32_matmul<
                               TensorType::BF16>(
                               swiglu,
                               weights->typed_data(),
                               output,
                               m,
                               n,
                               k,
                               alpha,
                               beta);
                }
                default:
                    return false;
                }
            }

            /**
             * @brief Validate the type recorded by either ownership constructor.
             *
             * A null tensor remains representable for legacy diagnostic callers;
             * execution methods reject it before dereference. Any non-null tensor
             * must use one of the floating formats implemented by this kernel.
             */
            void validateBoundWeight()
            {
                if (!weight_tensor_)
                    return;

                weight_type_ = weight_tensor_->native_type();
                if (weight_type_ != TensorType::FP32 &&
                    weight_type_ != TensorType::FP16 &&
                    weight_type_ != TensorType::BF16)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] Unsupported weight type: "
                              << static_cast<int>(weight_type_));
                    throw std::runtime_error(
                        "FloatingPointGemmKernel only supports FP32, FP16, or BF16 weights");
                }
            }

            /** Owns the exact contiguous bytes used by a prepared expert engine. */
            std::shared_ptr<const TensorBase> weight_tensor_lifetime_;
            const TensorBase *weight_tensor_ = nullptr; ///< Exact tensor used by GEMM.
            TensorType weight_type_ = TensorType::FP32;
            NumericalPolicy numerical_policy_ = NumericalPolicy::BackendNative;
        };

    } // namespace gemm
} // namespace llaminar2
