/**
 * @file RMSNormPrimitives.cpp
 * @brief CPU RMSNorm numerical leaves over original tensor storage.
 *
 * ISA-specific leaves retain their declared reduction/inverse/publication
 * programs. Native ARM consumes original FP32/FP16/BF16/INT32 values through
 * fixed register reductions and the canonical portable half codec. Native Q16
 * tiles retain separately rounded FP32 decode/squares before FP64 accumulation;
 * they share gamma without constructing decoded row banks. Caller-owned tensors, gamma and scratch retain their
 * lifetime/admission authority; synchronous arithmetic creates no new lease.
 * Generic FP32 rows bind a compiled scalar/AVX publisher before worksharing;
 * complete fragments and tails retain the same separately rounded products.
 * @author David Sanftenberg
 */

#include "RMSNormPrimitives.h"
#include "CPURMSNormBorrow.h"
#include "../../../tensors/FP16Utils.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include "CPUInt32RMSNormPublication.h"
#include "RMSNumericFatal.h"

#include "../../../utils/OpenMPUtils.h"
#include "../../../utils/CPUFeatures.h"
#include "../../../utils/Assertions.h"
#if defined(__aarch64__) && defined(__ARM_NEON)
#include "NEONRMSNormFloatingRows.h"
#include "NEONRMSNormQ16Rows.h"
#endif

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#include "../../../tensors/X86FP32.h"
#endif

namespace llaminar2::primitives
{

    /** @brief Expand one original half word through the canonical portable codec.
     * @param value Original FP16 word; @return Exact portable FP32 storage value.
     */
    __attribute__((always_inline)) inline float fp16_to_fp32_scalar(uint16_t value)
    {
        return ::llaminar2::fp16_to_fp32(value);
    }

    /** @brief Publish one half tail through the same normal RNE/payload contract as native ARM.
     * @param value Normalized FP32 value; @return Canonical FP16 storage word.
     */
    __attribute__((always_inline)) inline uint16_t fp32_to_fp16_scalar(float value)
    {
        return ::llaminar2::fp32_to_fp16(value);
    }

    // --- BF16 Scalar ---
    __attribute__((always_inline)) inline float bf16_to_fp32_scalar(uint16_t h)
    {
        uint32_t bits = static_cast<uint32_t>(h) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(float));
        return f;
    }

    __attribute__((always_inline)) inline uint16_t fp32_to_bf16_scalar(float f)
    {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(float));
        uint32_t lsb = (bits >> 16) & 1;
        uint32_t bias = 0x7FFF + lsb;
        bits += bias;
        return static_cast<uint16_t>(bits >> 16);
    }

#if defined(__AVX512F__)
    // --- AVX512 BF16 ---
    __attribute__((always_inline)) inline __m512 bf16_to_fp32_avx512(const uint16_t *bf16_ptr)
    {
        __m256i bf16_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(bf16_ptr));
        __m512i fp32_int = _mm512_cvtepu16_epi32(bf16_vec);
        __m512i shifted = _mm512_slli_epi32(fp32_int, 16);
        return _mm512_castsi512_ps(shifted);
    }

    __attribute__((always_inline)) inline void fp32_to_bf16_avx512(__m512 fp32_vec, uint16_t *bf16_ptr)
    {
        __m512i fp32_int = _mm512_castps_si512(fp32_vec);
        __m512i bits = fp32_int;
        __m512i bit16 = _mm512_srli_epi32(bits, 16);
        __m512i lsb = _mm512_and_si512(bit16, _mm512_set1_epi32(1));
        __m512i bias_base = _mm512_set1_epi32(0x7FFF);
        __m512i bias = _mm512_add_epi32(bias_base, lsb);
        __m512i rounded = _mm512_add_epi32(bits, bias);
        __m512i bf16_32 = _mm512_srli_epi32(rounded, 16);
        __m256i bf16_vec = _mm512_cvtepi32_epi16(bf16_32);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(bf16_ptr), bf16_vec);
    }

    // --- AVX512 FP16 ---
#if defined(__AVX512FP16__)
    __attribute__((always_inline)) inline __m512 fp16_to_fp32_avx512fp16(const uint16_t *fp16_ptr)
    {
        __m256h fp16_vec = _mm256_loadu_ph(reinterpret_cast<const void *>(fp16_ptr));
        return _mm512_cvtph_ps(fp16_vec);
    }

    __attribute__((always_inline)) inline void fp32_to_fp16_avx512fp16(__m512 fp32_vec, uint16_t *fp16_ptr)
    {
        __m256h fp16_vec = _mm512_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_ph(reinterpret_cast<void *>(fp16_ptr), fp16_vec);
    }
#else
    __attribute__((always_inline)) inline __m512 fp16_to_fp32_avx512(const uint16_t *fp16_ptr)
    {
        __m256i fp16_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(fp16_ptr));
        return _mm512_cvtph_ps(fp16_vec);
    }

    __attribute__((always_inline)) inline void fp32_to_fp16_avx512(__m512 fp32_vec, uint16_t *fp16_ptr)
    {
        __m256i fp16_vec = _mm512_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(fp16_ptr), fp16_vec);
    }
#endif
#endif

#if defined(__AVX2__)
    // --- AVX2 BF16 ---
    __attribute__((always_inline)) inline __m256 bf16_to_fp32_avx2(const uint16_t *bf16_ptr)
    {
        __m128i bf16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(bf16_ptr));
        __m256i fp32_int = _mm256_cvtepu16_epi32(bf16_vec);
        __m256i shifted = _mm256_slli_epi32(fp32_int, 16);
        return _mm256_castsi256_ps(shifted);
    }

    __attribute__((always_inline)) inline void fp32_to_bf16_avx2(__m256 fp32_vec, uint16_t *bf16_ptr)
    {
        __m256i fp32_int = _mm256_castps_si256(fp32_vec);
        __m256i bits = fp32_int;
        __m256i bit16 = _mm256_srli_epi32(bits, 16);
        __m256i lsb = _mm256_and_si256(bit16, _mm256_set1_epi32(1));
        __m256i bias_base = _mm256_set1_epi32(0x7FFF);
        __m256i bias = _mm256_add_epi32(bias_base, lsb);
        __m256i rounded = _mm256_add_epi32(bits, bias);
        __m256i bf16_32 = _mm256_srli_epi32(rounded, 16);
        __m256i packed = _mm256_packus_epi32(bf16_32, _mm256_setzero_si256());
        __m256i permuted = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
        __m128i result = _mm256_castsi256_si128(permuted);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(bf16_ptr), result);
    }

#if defined(__F16C__)
    // --- AVX2 FP16 ---
    __attribute__((always_inline)) inline __m256 fp16_to_fp32_avx2(const uint16_t *fp16_ptr)
    {
        __m128i fp16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(fp16_ptr));
        return _mm256_cvtph_ps(fp16_vec);
    }

    __attribute__((always_inline)) inline void fp32_to_fp16_avx2(__m256 fp32_vec, uint16_t *fp16_ptr)
    {
        __m128i fp16_vec = _mm256_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(fp16_ptr), fp16_vec);
    }
#endif
#endif

    namespace detail
    {
        inline bool want_parallel(std::size_t rows, std::size_t cols, const RMSNormExecOptions &opts)
        {
            if (!opts.allow_parallel)
                return false;

            std::size_t elems = rows * cols;

            // Single-row decode for large models (d_model >= 2048)
            // If rows=1, we want parallel if cols is large enough.
            if (rows == 1)
            {
                // Require significant work per thread to justify parallelization overhead.
                // For 3584 elements, sequential is likely faster than spawning threads.
                return cols >= 8192;
            }

            // NOTE: With OMP_WORKSHARE_REGION, we don't need to check omp_in_parallel()
            // here because the macro handles nested parallelism correctly - it will use
            // worksharing constructs when inside an existing parallel region.

            return elems >= opts.parallel_threshold_elems;
        }

#if defined(__AVX512F__)
        double compute_sumsq_avx512(const float *row, std::size_t cols)
        {
            double acc = 0.0;
            long long c = 0;
            __m512d dacc0 = _mm512_setzero_pd();
            __m512d dacc1 = _mm512_setzero_pd();
            __m512d dacc2 = _mm512_setzero_pd();
            __m512d dacc3 = _mm512_setzero_pd();

            for (; c + 64 <= (long long)cols; c += 64)
            {
                const float *base = row + c;

                // Load 4x 16-float blocks
                __m512 v0 = _mm512_loadu_ps(base + 0);
                __m512 v1 = _mm512_loadu_ps(base + 16);
                __m512 v2 = _mm512_loadu_ps(base + 32);
                __m512 v3 = _mm512_loadu_ps(base + 48);

                // Convert to double (split each 16-float into 2x 8-double)
                __m256 lo0 = _mm512_castps512_ps256(v0);
                __m256 hi0 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(v0), 1));
                __m256 lo1 = _mm512_castps512_ps256(v1);
                __m256 hi1 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(v1), 1));
                __m256 lo2 = _mm512_castps512_ps256(v2);
                __m256 hi2 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(v2), 1));
                __m256 lo3 = _mm512_castps512_ps256(v3);
                __m256 hi3 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(v3), 1));

                __m512d d0a = _mm512_cvtps_pd(lo0);
                __m512d d0b = _mm512_cvtps_pd(hi0);
                __m512d d1a = _mm512_cvtps_pd(lo1);
                __m512d d1b = _mm512_cvtps_pd(hi1);
                __m512d d2a = _mm512_cvtps_pd(lo2);
                __m512d d2b = _mm512_cvtps_pd(hi2);
                __m512d d3a = _mm512_cvtps_pd(lo3);
                __m512d d3b = _mm512_cvtps_pd(hi3);

                // FMA: acc += d * d
                // Distribute across 4 accumulators to break dependency chains
                dacc0 = _mm512_fmadd_pd(d0a, d0a, dacc0);
                dacc1 = _mm512_fmadd_pd(d0b, d0b, dacc1);
                dacc2 = _mm512_fmadd_pd(d1a, d1a, dacc2);
                dacc3 = _mm512_fmadd_pd(d1b, d1b, dacc3);

                dacc0 = _mm512_fmadd_pd(d2a, d2a, dacc0);
                dacc1 = _mm512_fmadd_pd(d2b, d2b, dacc1);
                dacc2 = _mm512_fmadd_pd(d3a, d3a, dacc2);
                dacc3 = _mm512_fmadd_pd(d3b, d3b, dacc3);
            }

            // Horizontal reduction
            acc += _mm512_reduce_add_pd(dacc0);
            acc += _mm512_reduce_add_pd(dacc1);
            acc += _mm512_reduce_add_pd(dacc2);
            acc += _mm512_reduce_add_pd(dacc3);

            // Tail: 16-float blocks
            for (; c + 16 <= (long long)cols; c += 16)
            {
                __m512 v = _mm512_loadu_ps(row + c);
                __m256 vlo = _mm512_castps512_ps256(v);
                __m256 vhi = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(v), 1));

                __m512d d0 = _mm512_cvtps_pd(vlo);
                __m512d d1 = _mm512_cvtps_pd(vhi);

                __m512d sq0 = _mm512_mul_pd(d0, d0);
                __m512d sq1 = _mm512_mul_pd(d1, d1);

                acc += _mm512_reduce_add_pd(sq0);
                acc += _mm512_reduce_add_pd(sq1);
            }

            // Scalar tail
            for (; c < (long long)cols; ++c)
            {
                double v = (double)row[c];
                acc += v * v;
            }
            return acc;
        }
#endif
#if defined(__AVX2__)
        double compute_sumsq_avx2(const float *row, std::size_t cols)
        {
            double acc = 0.0;
            long long c = 0;
            __m256d dacc0 = _mm256_setzero_pd();
            __m256d dacc1 = _mm256_setzero_pd();
            __m256d dacc2 = _mm256_setzero_pd();
            __m256d dacc3 = _mm256_setzero_pd();

            for (; c + 32 <= (long long)cols; c += 32)
            {
                const float *base = row + c;
                __m256 v0 = _mm256_loadu_ps(base + 0);
                __m256 v1 = _mm256_loadu_ps(base + 8);
                __m256 v2 = _mm256_loadu_ps(base + 16);
                __m256 v3 = _mm256_loadu_ps(base + 24);

                // Convert v0
                __m128 v0_lo = _mm256_castps256_ps128(v0);
                __m128 v0_hi = _mm256_extractf128_ps(v0, 1);
                __m256d d0_lo = _mm256_cvtps_pd(v0_lo);
                __m256d d0_hi = _mm256_cvtps_pd(v0_hi);

                // Convert v1
                __m128 v1_lo = _mm256_castps256_ps128(v1);
                __m128 v1_hi = _mm256_extractf128_ps(v1, 1);
                __m256d d1_lo = _mm256_cvtps_pd(v1_lo);
                __m256d d1_hi = _mm256_cvtps_pd(v1_hi);

                // Convert v2
                __m128 v2_lo = _mm256_castps256_ps128(v2);
                __m128 v2_hi = _mm256_extractf128_ps(v2, 1);
                __m256d d2_lo = _mm256_cvtps_pd(v2_lo);
                __m256d d2_hi = _mm256_cvtps_pd(v2_hi);

                // Convert v3
                __m128 v3_lo = _mm256_castps256_ps128(v3);
                __m128 v3_hi = _mm256_extractf128_ps(v3, 1);
                __m256d d3_lo = _mm256_cvtps_pd(v3_lo);
                __m256d d3_hi = _mm256_cvtps_pd(v3_hi);

                dacc0 = _mm256_fmadd_pd(d0_lo, d0_lo, dacc0);
                dacc1 = _mm256_fmadd_pd(d0_hi, d0_hi, dacc1);
                dacc2 = _mm256_fmadd_pd(d1_lo, d1_lo, dacc2);
                dacc3 = _mm256_fmadd_pd(d1_hi, d1_hi, dacc3);

                dacc0 = _mm256_fmadd_pd(d2_lo, d2_lo, dacc0);
                dacc1 = _mm256_fmadd_pd(d2_hi, d2_hi, dacc1);
                dacc2 = _mm256_fmadd_pd(d3_lo, d3_lo, dacc2);
                dacc3 = _mm256_fmadd_pd(d3_hi, d3_hi, dacc3);
            }

            // Combine accumulators
            __m256d dacc01 = _mm256_add_pd(dacc0, dacc1);
            __m256d dacc23 = _mm256_add_pd(dacc2, dacc3);
            __m256d dacc_all = _mm256_add_pd(dacc01, dacc23);

            // Horizontal reduction
            __m128d lo = _mm256_castpd256_pd128(dacc_all);
            __m128d hi = _mm256_extractf128_pd(dacc_all, 1);
            __m128d sum2 = _mm_add_pd(lo, hi);
            __m128d swap = _mm_permute_pd(sum2, 1);
            __m128d sum1 = _mm_add_pd(sum2, swap);

            double partial;
            _mm_store_sd(&partial, sum1);
            acc += partial;

            // 8-element blocks
            for (; c + 8 <= (long long)cols; c += 8)
            {
                __m256 v = _mm256_loadu_ps(row + c);
                __m128 v_lo = _mm256_castps256_ps128(v);
                __m128 v_hi = _mm256_extractf128_ps(v, 1);

                __m256d d_lo = _mm256_cvtps_pd(v_lo);
                __m256d d_hi = _mm256_cvtps_pd(v_hi);

                __m256d sq_lo = _mm256_mul_pd(d_lo, d_lo);
                __m256d sq_hi = _mm256_mul_pd(d_hi, d_hi);

                __m256d sum_blk = _mm256_add_pd(sq_lo, sq_hi);

                __m128d l = _mm256_castpd256_pd128(sum_blk);
                __m128d h = _mm256_extractf128_pd(sum_blk, 1);
                __m128d s = _mm_add_pd(l, h);
                __m128d sw = _mm_permute_pd(s, 1);
                __m128d s1 = _mm_add_pd(s, sw);

                double p;
                _mm_store_sd(&p, s1);
                acc += p;
            }

            // Scalar tail
            for (; c < (long long)cols; ++c)
            {
                double v = (double)row[c];
                acc += v * v;
            }
            return acc;
        }
#endif

        /** @brief Reduce original FP32 words in increasing-column FP64 FMA order.
         * @param row Original contiguous words; may be null only for an empty row.
         * @param cols Original logical width; @return Explicit scalar squared summary.
         * @details A finite FP32 square is exact in FP64. The dependency chain
         * fixes summation order independently of vector width and instrumentation.
         */
        double compute_sumsq_scalar(const float *row, std::size_t cols)
        {
            double sum = 0.0;
            for (std::size_t c = 0; c < cols; ++c)
            {
                const double value = double(row[c]);
                sum = std::fma(value, value, sum);
            }
            return sum;
        }

#if defined(__aarch64__) && defined(__ARM_NEON)
        /** @brief Reduce one original FP32 row with the native fixed FP64 tree.
         * @param row Original synchronous row borrow; @param cols Logical count.
         * @return Native FP64 squared summary, independent of M/team partition.
         */
        double compute_sumsq_neon(const float *row, std::size_t cols)
        {
            using namespace neon_rmsnorm;
            return squaredSums(OriginalRows<Storage::FP32, 1>::bind({std::span<const float>(row, cols)}, cols))[0];
        }
#endif

        /** @brief Original gamma authorship, resolved before entering a worker region. */
        enum class FP32Gamma { Identity, Original };
        /** @brief One completed-row publisher over already checked original spans. */
        using FP32RowApplication = void (*)(std::span<const float>,std::span<const float>,std::span<float>,float);

        /** @brief Publish original FP32 columns with explicit compiled fragments and bounded tails.
         * @tparam Program Actual compiled row-application ISA, never an absent-ISA alias.
         * @tparam Gamma Original weights or declared identity.
         * @param input Complete original row, consumed after its summary is finished.
         * @param gamma Immutable original weights; empty only for identity.
         * @param output Original row destination; the caller seals alias/range policy.
         * @param inverse Completed private FP32 inverse.
         * @details Each column separately rounds input*inverse before multiplying
         * original gamma. Vectorization changes no arithmetic edge or reduction.
         */
        template<ISALevel Program,FP32Gamma Gamma>
        void apply_fp32_row(std::span<const float>input,std::span<const float>gamma,
                            std::span<float>output,float inverse)
        {
            static_assert(Program==ISALevel::Scalar
#if defined(__AVX2__)
                ||Program==ISALevel::AVX2
#endif
#if defined(__AVX512F__)
                ||Program==ISALevel::AVX512
#endif
            );
            std::size_t column=0;
#if defined(__AVX512F__)
            if constexpr(Program==ISALevel::AVX512)
            {
                const auto factor=_mm512_set1_ps(inverse);
                for(;input.size()-column>=16;column+=16)
                {
                    auto value=_mm512_mul_ps(_mm512_loadu_ps(input.data()+column),factor);
                    if constexpr(Gamma==FP32Gamma::Original)
                        value=_mm512_mul_ps(value,_mm512_loadu_ps(gamma.data()+column));
                    _mm512_storeu_ps(output.data()+column,value);
                }
            }
#endif
#if defined(__AVX2__)
            if constexpr(Program==ISALevel::AVX2)
            {
                const auto factor=_mm256_set1_ps(inverse);
                for(;input.size()-column>=8;column+=8)
                {
                    auto value=_mm256_mul_ps(_mm256_loadu_ps(input.data()+column),factor);
                    if constexpr(Gamma==FP32Gamma::Original)
                        value=_mm256_mul_ps(value,_mm256_loadu_ps(gamma.data()+column));
                    _mm256_storeu_ps(output.data()+column,value);
                }
            }
#endif
            for(;column<input.size();++column)
            {
                const float scaled=input[column]*inverse;
                if constexpr(Gamma==FP32Gamma::Original)output[column]=scaled*gamma[column];
                else output[column]=scaled;
            }
        }

        /** @brief Bind only an actual scalar program or an explicitly compiled native publisher.
         * @tparam Gamma Already resolved original gamma authorship.
         * @param isa Authored scalar intent or the actual active native ISA.
         * @return Persistent function identity for this synchronous invocation.
         * @details Unsupported ISA requests are fatal before worksharing; no
         * scalar substitute or forced-vectorizer promise conceals a missing leaf.
         */
        template<FP32Gamma Gamma>
        FP32RowApplication fp32_row_application(ISALevel isa)
        {
            switch(isa)
            {
            case ISALevel::Scalar:return &apply_fp32_row<ISALevel::Scalar,Gamma>;
#if defined(__AVX2__)
            case ISALevel::AVX2:return &apply_fp32_row<ISALevel::AVX2,Gamma>;
#endif
#if defined(__AVX512F__)
            case ISALevel::AVX512:return &apply_fp32_row<ISALevel::AVX512,Gamma>;
#endif
            default:
                std::fputs("ERROR: CPU FP32 RMS publication received an uncompiled ISA\n",stderr);
                std::abort();
            }
        }

        __attribute__((always_inline)) inline double compute_sumsq_dispatch(const float *row, std::size_t cols)
        {
            return ISA_DISPATCH_RETVAL(compute_sumsq, row, cols);
        }


#if defined(__aarch64__) && defined(__ARM_NEON)
        /** @brief Execute one shared-gamma original floating tile without serial-call replay.
         * @tparam Encoding Exact original floating storage.
         * @tparam Count Physical rows1..4 participating in this tile.
         * @param bank Complete validated original bank.
         * @param first Original first row, never a worker-local row coordinate.
         * @param program Explicit FP64 or T5 FP32 arithmetic policy.
         */
        template<neon_rmsnorm::Storage Encoding, std::size_t Count>
        void native_floating_tile(const rmsnorm_operands::FloatingBorrow<neon_rmsnorm::Unit<Encoding>> &bank,
                                  std::size_t first, neon_rmsnorm::Accumulation program)
        {
            using namespace neon_rmsnorm;
            std::array<std::span<const Unit<Encoding>>, Count> input;
            std::array<std::span<Unit<Encoding>>, Count> output;
            neon_rmsnorm::detail::visit<Count>([&]<std::size_t Row> {
                input[Row] = bank.sourceRow(first + Row);
                output[Row] = bank.outputRow(first + Row);
            });
            normalize(OriginalRows<Encoding, Count>::bind(input, bank.columns()),
                      output, bank.gamma(), bank.epsilon(), program);
        }

        /** @brief Workshare complete native floating tiles, retaining one arithmetic program per row.
         * @tparam Encoding Exact original/result floating storage.
         * @param bank Complete checked original banks and policy, with a retained external lease.
         * @param opts Original execution options, not a second mutable scheduler authority.
         */
        template<neon_rmsnorm::Storage Encoding>
        void native_floating_matrix(const rmsnorm_operands::FloatingBorrow<neon_rmsnorm::Unit<Encoding>> &bank,
                                    const RMSNormExecOptions &opts)
        {
            using namespace neon_rmsnorm;
            const auto program = opts.t5_compat_mode ? Accumulation::T5FP32 : Accumulation::FP64;
            const auto tiles = bank.rows()/4 + (bank.rows()%4 != 0);
            auto work = [&]() {
#pragma omp for schedule(static)
                for (std::ptrdiff_t tile = 0; tile < std::ptrdiff_t(tiles); ++tile)
                {
                    const auto first = std::size_t(tile)*4;
                    switch (std::min<std::size_t>(4,bank.rows()-first))
                    {
                    case 4: native_floating_tile<Encoding,4>(bank,first,program); break;
                    case 3: native_floating_tile<Encoding,3>(bank,first,program); break;
                    case 2: native_floating_tile<Encoding,2>(bank,first,program); break;
                    case 1: native_floating_tile<Encoding,1>(bank,first,program); break;
                    }
                }
            };
            // Retained participants must all enter, including small groups for
            // which a standalone fork/join is not economical.
            OMP_WORKSHARE_REGION_IF(work, omp_in_parallel() ||
                (tiles > 1 && want_parallel(bank.rows(),bank.columns(),opts)));
        }

        /** @brief Workshare complete original Q16 tiles without row replay or decoded storage.
         * @param bank Checked compressed/expansion banks and original policy.
         * @param opts Retained execution intent; no new workspace or owner is created.
         */
        void native_q16_matrix(const rmsnorm_operands::Q16ExpansionBorrow &bank,const RMSNormExecOptions &opts)
        {
            const auto tiles=bank.empty()?0:bank.rows()/4+(bank.rows()%4!=0);
            auto work=[&](){
#pragma omp for schedule(static)
                for(std::ptrdiff_t tile=0;tile<std::ptrdiff_t(tiles);++tile)
                {
                    const auto first=std::size_t(tile)*4;
                    switch(std::min<std::size_t>(4,bank.rows()-first))
                    {
                    case 4: neon_rmsnorm::normalizeQ16<4>(bank,first);break;
                    case 3: neon_rmsnorm::normalizeQ16<3>(bank,first);break;
                    case 2: neon_rmsnorm::normalizeQ16<2>(bank,first);break;
                    case 1: neon_rmsnorm::normalizeQ16<1>(bank,first);break;
                    }
                }
            };
            // Retained small/empty invocations are still collective completion
            // boundaries, even when a standalone fork would not repay its cost.
            OMP_WORKSHARE_REGION_IF(work,omp_in_parallel()||
                (tiles>1&&want_parallel(bank.rows(),bank.columns(),opts)));
        }

        /** @brief Authenticate explicit native dispatch; a scalar request is diagnostic intent.
         * @param opts Original execution intent.
         * @return Whether this request names the available native ARM program.
         */
        bool use_native_rmsnorm(const RMSNormExecOptions &opts)
        {
            if (opts.force_scalar) return false;
            const auto isa = activeISALevel();
            return isa == ISALevel::NEON || isa == ISALevel::NEONDotProd;
        }
#endif
    } // namespace detail
    using namespace detail;

    void rmsnorm_compute_inv(
        const double *row_sumsq,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        float *inv_out)
    {
        if (!row_sumsq || !inv_out || rows == 0)
            return;

        double denom = (double)cols;
        if (denom <= 0.0)
            denom = 1.0;

        for (std::size_t r = 0; r < rows; ++r)
        {
            if (!std::isfinite(row_sumsq[r]))
                rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteSummary,
                    "CPU RMS observed a nonfinite original-row squared summary");
            double val = row_sumsq[r] / denom + (double)epsilon;
            double inv = (val > 0.0) ? 1.0 / std::sqrt(val) : 0.0;
            const float rounded = (float)inv;
            // A finite FP64 inverse may still round to FP32 infinity (tiny rows);
            // the published FP32 value must be representable before any store.
            if (!std::isfinite(rounded))
                rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteInverse,
                    "CPU RMS FP32 inverse is not representable after the declared rounding");
            inv_out[r] = rounded;
        }
    }


    void rmsnorm_fused_vectorized(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts)
    {
        const auto bank = rmsnorm_operands::FloatingBorrow<float>::bind(src,gamma,dst,rows,cols,epsilon);
        bank.requireMetadataReadOnly(&opts,sizeof(opts));
        if (!::llaminar::v2::beginWorkshare(!bank.empty())) return;
#if defined(__aarch64__) && defined(__ARM_NEON)
        if (use_native_rmsnorm(opts))
        {
            native_floating_matrix<neon_rmsnorm::Storage::FP32>(bank,opts);
            return;
        }
#endif
        const auto publicationISA=opts.force_scalar?ISALevel::Scalar:activeISALevel();
        const auto application=bank.gamma().empty()?fp32_row_application<FP32Gamma::Identity>(publicationISA):
                                                   fp32_row_application<FP32Gamma::Original>(publicationISA);
        // A complete row, its summary and inverse have one elected worker.
        // No TLS matrix, worker-partitioned arithmetic, atomics or detached
        // summary/application phases survive. This also seals the x86 M=1
        // single-worker program across all M and retained-team sizes.
        auto work = [&]() {
#pragma omp for schedule(static)
            for (std::ptrdiff_t r=0;r<std::ptrdiff_t(bank.rows());++r)
            {
                const auto input=bank.sourceRow(std::size_t(r));
                const auto output=bank.outputRow(std::size_t(r));
                double sum;
                if (opts.t5_compat_mode)
                {
                    float partial=0.0f;
                    for(float value:input) partial=std::fma(value,value,partial);
                    sum=double(partial);
                }
                else sum=opts.force_scalar?compute_sumsq_scalar(input.data(),cols):
                                            compute_sumsq_dispatch(input.data(),cols);
                if (!std::isfinite(sum))
                    rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteSummary,
                        "CPU RMS observed a nonfinite original-row squared summary");
                float inverse;
                rmsnorm_compute_inv(&sum,1,cols,epsilon,&inverse);
                if (inverse==0.0f) std::fill(output.begin(),output.end(),0.0f);
                else application(input,bank.gamma(),output,inverse);
            }
        };
        OMP_WORKSHARE_REGION_IF(work,omp_in_parallel() || (rows>1 && want_parallel(rows,cols,opts)));
    }

    // ========================================================================
    // INT32 RMSNorm Implementation (for full INT8 pipelines)
    // ========================================================================

    namespace detail
    {
        /** @brief Reduce full-width original integers in the declared scalar FP64 program.
         * @param row Original integer words; @param cols Logical width.
         * @return Increasing-column FP64 FMA sum, without a lossy FP32 conversion.
         */
        double compute_rms_sq_int32_scalar(const int32_t *row, std::size_t cols)
        {
            double acc = 0.0;
            for (std::size_t c = 0; c < cols; ++c)
            {
                const double value=double(row[c]);
                acc=std::fma(value,value,acc);
            }
            return acc;
        }

#if defined(__AVX2__)
        /** @brief Reduce original INT32 through eight fixed FP64 chains, never FP32.
         * @param row Complete original integer row; @param cols Logical width.
         * @return Fixed eight-chain summary with bounded masked original tails.
         * @details Integer-to-double conversion retains low bits above2^24.
         * The fold is ((0+2)+(4+6))+((1+3)+(5+7)); no worker partition enters it.
         */
        double compute_rms_sq_int32_avx2(const int32_t *row, std::size_t cols)
        {
            auto low=_mm256_setzero_pd(),high=_mm256_setzero_pd();
            const auto fragment=[](const int32_t*original,int live,__m256d &sum)
            {
                const auto mask=_mm_setr_epi32(live>0?-1:0,live>1?-1:0,live>2?-1:0,live>3?-1:0);
                const auto words=live==4?_mm_loadu_si128(reinterpret_cast<const __m128i*>(original)):
                    _mm_maskload_epi32(original,mask);
                const auto values=_mm256_cvtepi32_pd(words);
                sum=_mm256_fmadd_pd(values,values,sum);
            };
            std::size_t column=0;
            for(;cols-column>=8;column+=8)
            {
                fragment(row+column,4,low);fragment(row+column+4,4,high);
            }
            if(column<cols)
            {
                fragment(row+column,int(std::min<std::size_t>(4,cols-column)),low);
                if(cols-column>4)fragment(row+column+4,int(cols-column-4),high);
            }
            const auto first=_mm_add_pd(_mm256_castpd256_pd128(low),_mm256_extractf128_pd(low,1));
            const auto second=_mm_add_pd(_mm256_castpd256_pd128(high),_mm256_extractf128_pd(high,1));
            const auto folded=_mm_add_pd(first,second);
            return _mm_cvtsd_f64(_mm_add_sd(folded,_mm_unpackhi_pd(folded,folded)));
        }
#endif

#if defined(__AVX512F__)
        /** @brief Retain the existing sixteen-chain AVX512 FP64 integer summary.
         * @param row Complete original signed integers; @param cols Logical width.
         * @return Existing AVX512 reduction tree; not a universal cross-ISA oracle.
         */
        double compute_rms_sq_int32_avx512(const int32_t *row, std::size_t cols)
        {
            double acc = 0.0;
            long long c = 0;
            __m512d dacc0 = _mm512_setzero_pd();
            __m512d dacc1 = _mm512_setzero_pd();

            for (; c + 16 <= (long long)cols; c += 16)
            {
                __m512i i32_lo = _mm512_loadu_si512((__m512i *)(row + c));
                __m256i i32_lo_256 = _mm512_castsi512_si256(i32_lo);
                __m256i i32_hi_256 = _mm512_extracti64x4_epi64(i32_lo, 1);
                __m512d d0 = _mm512_cvtepi32_pd(i32_lo_256);
                __m512d d1 = _mm512_cvtepi32_pd(i32_hi_256);
                dacc0 = _mm512_fmadd_pd(d0, d0, dacc0);
                dacc1 = _mm512_fmadd_pd(d1, d1, dacc1);
            }

            acc += _mm512_reduce_add_pd(dacc0);
            acc += _mm512_reduce_add_pd(dacc1);

            for (; c + 8 <= (long long)cols; c += 8)
            {
                __m256i i32 = _mm256_loadu_si256((__m256i *)(row + c));
                __m512d d = _mm512_cvtepi32_pd(i32);
                __m512d sq = _mm512_mul_pd(d, d);
                acc += _mm512_reduce_add_pd(sq);
            }

            for (; c < (long long)cols; ++c)
            {
                double v = (double)row[c];
                acc += v * v;
            }
            return acc;
        }
#endif

#if LLAMINAR_COMPILED_WITH_NEON
        /** @brief Share original integer traversal across one through four summary rows.
         * @tparam Rows Live original rows, not padded allocation capacity.
         * @tparam Bank Typed complete original summary or fused-publication borrow.
         * @param bank Complete checked original input/output; @param first Original first row.
         * @return Private fixed-order FP64 summaries, not converted original rows.
         * @details Every row retains its native eight-chain M1 arithmetic. Only
         * pointer descriptors are assembled; no integer/FP32 row is copied.
         */
        template<std::size_t Rows,class Bank>
        std::array<double,Rows> native_int32_summaries(const Bank &bank,std::size_t first)
        {
            std::array<std::span<const std::int32_t>,Rows>inputs;
            for(std::size_t row=0;row<Rows;++row)inputs[row]=bank.sourceRow(first+row);
            return neon_rmsnorm::squaredSums(
                neon_rmsnorm::OriginalRows<neon_rmsnorm::Storage::Int32,Rows>::bind(inputs,bank.columns()));
        }
#endif
        /** @brief Resolve explicit scalar intent or one complete native integer RMS program.
         * @param options Immutable execution policy; @return Authenticated ISA identity.
         * @throws std::invalid_argument For undeclared T5 or a foreign/uncompiled native program.
         */
        ISALevel int32_rms_program(const RMSNormExecOptions &options)
        {
            if(options.t5_compat_mode)rms_numeric::rejectPolicy("INT32 RMS has no declared T5 FP32 accumulation program");
            const auto isa=options.force_scalar?ISALevel::Scalar:activeISALevel();
            if(isa!=ISALevel::Scalar&&!isNativeCPUExecutionISAPair(compiledCPUISALevel(),isa))
                rms_numeric::rejectPolicy("INT32 RMS has no matching compiled native ISA");
            return isa;
        }
        /** @brief Reduce one original row under an already admitted non-NEON program.
         * @param row Complete original integers; @param isa Explicit scalar or compiled x86 program.
         * @return Complete fixed-tree FP64 summary, without decoded source storage.
         */
        double int32_rms_summary(std::span<const std::int32_t>row,ISALevel isa)
        {
            switch(isa)
            {
            case ISALevel::Scalar:return compute_rms_sq_int32_scalar(row.data(),row.size());
#if defined(__AVX2__)
            case ISALevel::AVX2:return compute_rms_sq_int32_avx2(row.data(),row.size());
#endif
#if defined(__AVX512F__)
            case ISALevel::AVX512:return compute_rms_sq_int32_avx512(row.data(),row.size());
#endif
            default:LLAMINAR_UNREACHABLE("INT32 RMS summary selected an uncompiled instruction program");
            }
        }

    } // namespace detail
    using namespace detail;

    /** @brief Publish complete full-width INT32 summaries through one lexical team owner.
     * @param src Original integer bank; @param rows Logical rows; @param cols Original width.
     * @param row_sumsq Disjoint caller-owned FP64 summaries; @param opts Declared execution/policy.
     * @throws std::invalid_argument For invalid geometry, source/policy aliases or undeclared T5 intent.
     * @details Scalar intent is increasing-K FP64 FMA; native arithmetic retains
     * its declared per-ISA tree. Empty and small retained calls are collective
     * completion edges, never independent whole-bank replays by every member.
     */
    void rmsnorm_compute_row_sumsq_int32_vectorized(
        const int32_t *src,
        std::size_t rows,
        std::size_t cols,
        double *row_sumsq,
        const RMSNormExecOptions &opts)
    {
        const auto bank=rmsnorm_operands::Int32ReductionBorrow::bind(src,row_sumsq,rows,cols);
        bank.requireMetadataReadOnly(&opts,sizeof(opts));
        const auto isa=int32_rms_program(opts);
        if(!::llaminar::v2::beginWorkshare(!bank.empty()))return;
        const auto tiles=rows/4+(rows%4!=0);
        auto work=[&]()
        {
#pragma omp for schedule(static)
            for(std::ptrdiff_t tile=0;tile<std::ptrdiff_t(tiles);++tile)
            {
                const auto first=std::size_t(tile)*4;
                const auto live=std::min<std::size_t>(4,rows-first);
#if LLAMINAR_COMPILED_WITH_NEON
                if(isa==ISALevel::NEON||isa==ISALevel::NEONDotProd)
                {
                    const auto publish=[&]<std::size_t Count>()
                    {
                        const auto sums=native_int32_summaries<Count>(bank,first);
                        for(std::size_t row=0;row<Count;++row)bank.publish(first+row,sums[row]);
                    };
                    switch(live)
                    {
                    case 1:publish.template operator()<1>();break;
                    case 2:publish.template operator()<2>();break;
                    case 3:publish.template operator()<3>();break;
                    case 4:publish.template operator()<4>();break;
                    default:LLAMINAR_UNREACHABLE("INT32 RMS original tile lost its live rows");
                    }
                    continue;
                }
#endif
                for(std::size_t row=first;row<first+live;++row)
                {
                    bank.publish(row,int32_rms_summary(bank.sourceRow(row),isa));
                }
            }
        };
        OMP_WORKSHARE_REGION_IF(work,omp_in_parallel()||(tiles>1&&want_parallel(rows,cols,opts)));
    }

    /** @brief Normalize/requantize original integer rows without a converted TLS bank.
     * @param src Original integer bank; @param gamma Optional immutable FP32 weights.
     * @param inv Complete immutable FP32 row inverses; @param rows Original rows.
     * @param cols Original width; @param dst_int8 Disjoint signed-code publication.
     * @param dst_row_scales Disjoint row metadata; @param opts Explicit ISA/execution policy.
     * @throws std::invalid_argument For malformed geometry, aliases or undeclared policy.
     * @details Each worker owns its row's maximum, recomputed products, codes and
     * scale. The native row-INT8 program has no block-Q8/FP32-quantizer floor.
     */
    void rmsnorm_apply_int32_to_int8_vectorized(
        const int32_t *src,const float *gamma,const float *inv,std::size_t rows,std::size_t cols,
        int8_t *dst_int8,float *dst_row_scales,const RMSNormExecOptions &opts)
    {
        const auto bank=rmsnorm_operands::Int32QuantizationBorrow::bind(src,gamma,dst_int8,dst_row_scales,rows,cols);
        bank.requireMetadataReadOnly(&opts,sizeof(opts));
        rmsnorm_operands::requireBank(inv,bank.empty()?0:rows,sizeof(float));
        bank.requireMetadataReadOnly(inv,bank.empty()?0:rows*sizeof(float));
        const auto isa=int32_rms_program(opts);
        if(!::llaminar::v2::beginWorkshare(!bank.empty()))return;
        auto work=[&]()
        {
#pragma omp for schedule(static)
            for(std::ptrdiff_t row=0;row<std::ptrdiff_t(rows);++row)
                int32_rms_publication::publishRow(bank.row(std::size_t(row)),inv[std::size_t(row)],isa);
        };
        OMP_WORKSHARE_REGION_IF(work,omp_in_parallel()||(rows>1&&want_parallel(rows,cols,opts)));
    }

    /** @brief Fuse original integer summary/inverse/maximum/publication under one tile owner.
     * @param src Original complete integer bank; @param gamma Optional immutable gamma.
     * @param dst_int8 Exclusive signed-code output; @param dst_row_scales Exclusive row scales.
     * @param rows Original row count; @param cols Original logical width.
     * @param epsilon Finite nonnegative staged FP64-mean policy; @param opts Explicit program/execution intent.
     * @throws std::invalid_argument For malformed geometry/policy or writable intersections.
     * @details Native ARM shares the fixed summary traversal across1..4 original
     * rows. Inverse retains double(sum)/cols+double(epsilon), then one final
     * float conversion; it is not the separate per-row FP32-mean program.
     * Original values are streamed twice for publication, never retained as a
     * decoded row. No scratch resizing, private summary bank or row replay runs.
     */
    void rmsnorm_fused_int32_to_int8_vectorized(
        const int32_t *src,const float *gamma,int8_t *dst_int8,float *dst_row_scales,
        std::size_t rows,std::size_t cols,float epsilon,const RMSNormExecOptions &opts)
    {
        if(!std::isfinite(epsilon)||epsilon<0.0f)
            rms_numeric::rejectPolicy("INT32 RMS requires finite nonnegative original epsilon");
        const auto bank=rmsnorm_operands::Int32QuantizationBorrow::bind(src,gamma,dst_int8,dst_row_scales,rows,cols);
        bank.requireMetadataReadOnly(&opts,sizeof(opts));
        const auto isa=int32_rms_program(opts);
        if(!::llaminar::v2::beginWorkshare(!bank.empty()))return;
        const auto tiles=rows/4+(rows%4!=0);
        auto work=[&]()
        {
#pragma omp for schedule(static)
            for(std::ptrdiff_t tile=0;tile<std::ptrdiff_t(tiles);++tile)
            {
                const auto first=std::size_t(tile)*4,live=std::min<std::size_t>(4,rows-first);
                const auto normalize=[&]<std::size_t Count>()
                {
                    std::array<double,Count>sums;
#if LLAMINAR_COMPILED_WITH_NEON
                    if(isa==ISALevel::NEON||isa==ISALevel::NEONDotProd)
                        sums=native_int32_summaries<Count>(bank,first);
                    else
#endif
                        for(std::size_t row=0;row<Count;++row)sums[row]=int32_rms_summary(bank.sourceRow(first+row),isa);
                    for(std::size_t row=0;row<Count;++row)
                    {
                        const double mean=sums[row]/double(cols)+double(epsilon);
                        const float inverse=mean>0.0?float(1.0/std::sqrt(mean)):0.0f;
                        int32_rms_publication::publishRow(bank.row(first+row),inverse,isa);
                    }
                };
                switch(live)
                {
                case 1:normalize.template operator()<1>();break;
                case 2:normalize.template operator()<2>();break;
                case 3:normalize.template operator()<3>();break;
                case 4:normalize.template operator()<4>();break;
                default:LLAMINAR_UNREACHABLE("INT32 RMS fused original tile lost its live rows");
                }
            }
        };
        OMP_WORKSHARE_REGION_IF(work,omp_in_parallel()||(tiles>1&&want_parallel(rows,cols,opts)));
    }

    // ========================================================================
    // FP32 Per-Row Primitives (Testable SIMD variants)
    // ========================================================================

#if defined(__aarch64__) && defined(__ARM_NEON)
    void rmsnorm_fused_row_neon(const float *src,const float *gamma,float *dst,
                               std::size_t cols,float epsilon)
    {
        using namespace neon_rmsnorm;
        const auto bank=rmsnorm_operands::FloatingBorrow<float>::bind(src,gamma,dst,1,cols,epsilon);
        normalize(OriginalRows<Storage::FP32,1>::bind({bank.sourceRow(0)},cols),
                  std::array{bank.outputRow(0)},bank.gamma(),epsilon,
                  Accumulation::FP64,Inverse::FP32Mean);
    }
#endif

    void rmsnorm_fused_row_scalar(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t cols,
        float epsilon)
    {
        if (cols == 0) return; // Zero-width invocation is a no-op: no values or inverse.
        if (!std::isfinite(epsilon) || epsilon < 0.0f)
            rms_numeric::rejectPolicy("CPU FP32 RMS requires a finite nonnegative original epsilon");
        // Phase 1: Compute sum of squares (double precision)
        double sum_sq = 0.0;
        for (std::size_t c = 0; c < cols; ++c)
        {
            double val = static_cast<double>(src[c]);
            sum_sq += val * val;
        }
        if (!std::isfinite(sum_sq))
            rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteSummary,
                "CPU FP32 RMS observed a nonfinite original-row squared summary");

        // Phase 2: Compute inverse RMS (original FP32-mean program)
        const float denominator = static_cast<float>(sum_sq / cols) + epsilon;
        const float inv_rms = 1.0f / std::sqrt(denominator);
        if (!std::isfinite(inv_rms))
            rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteInverse,
                "CPU FP32 RMS original-row inverse is not representable");

        // Phase 3: Apply normalization; null gamma is admitted identity.
        if (gamma)
        {
            for (std::size_t c = 0; c < cols; ++c)
            {
                dst[c] = src[c] * inv_rms * gamma[c];
            }
        }
        else
        {
            for (std::size_t c = 0; c < cols; ++c)
            {
                dst[c] = src[c] * inv_rms;
            }
        }
    }

#if defined(__AVX2__)
    void rmsnorm_fused_row_avx2(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t cols,
        float epsilon)
    {
        if (cols == 0) return; // Zero-width invocation is a no-op: no values or inverse.
        if (!std::isfinite(epsilon) || epsilon < 0.0f)
            rms_numeric::rejectPolicy("CPU FP32 RMS requires a finite nonnegative original epsilon");
        // Phase 1: Compute sum of squares with AVX2
        double sum_sq = 0.0;
        std::size_t c = 0;

        // Multi-accumulator AVX2: process 64 elements per iteration
        __m256d dacc0 = _mm256_setzero_pd();
        __m256d dacc1 = _mm256_setzero_pd();
        __m256d dacc2 = _mm256_setzero_pd();
        __m256d dacc3 = _mm256_setzero_pd();

        for (; c + 64 <= cols; c += 64)
        {
            const float *base = src + c;

            // Load 4x 16-float blocks
            __m256 v0 = _mm256_loadu_ps(base + 0);
            __m256 v1 = _mm256_loadu_ps(base + 8);
            __m256 v2 = _mm256_loadu_ps(base + 16);
            __m256 v3 = _mm256_loadu_ps(base + 24);
            __m256 v4 = _mm256_loadu_ps(base + 32);
            __m256 v5 = _mm256_loadu_ps(base + 40);
            __m256 v6 = _mm256_loadu_ps(base + 48);
            __m256 v7 = _mm256_loadu_ps(base + 56);

            // Convert to double (split each 8-float into 2x 4-double)
            __m128 lo0 = _mm256_castps256_ps128(v0);
            __m128 hi0 = _mm256_extractf128_ps(v0, 1);
            __m128 lo1 = _mm256_castps256_ps128(v1);
            __m128 hi1 = _mm256_extractf128_ps(v1, 1);
            __m128 lo2 = _mm256_castps256_ps128(v2);
            __m128 hi2 = _mm256_extractf128_ps(v2, 1);
            __m128 lo3 = _mm256_castps256_ps128(v3);
            __m128 hi3 = _mm256_extractf128_ps(v3, 1);
            __m128 lo4 = _mm256_castps256_ps128(v4);
            __m128 hi4 = _mm256_extractf128_ps(v4, 1);
            __m128 lo5 = _mm256_castps256_ps128(v5);
            __m128 hi5 = _mm256_extractf128_ps(v5, 1);
            __m128 lo6 = _mm256_castps256_ps128(v6);
            __m128 hi6 = _mm256_extractf128_ps(v6, 1);
            __m128 lo7 = _mm256_castps256_ps128(v7);
            __m128 hi7 = _mm256_extractf128_ps(v7, 1);

            __m256d d0 = _mm256_cvtps_pd(lo0);
            __m256d d1 = _mm256_cvtps_pd(hi0);
            __m256d d2 = _mm256_cvtps_pd(lo1);
            __m256d d3 = _mm256_cvtps_pd(hi1);
            __m256d d4 = _mm256_cvtps_pd(lo2);
            __m256d d5 = _mm256_cvtps_pd(hi2);
            __m256d d6 = _mm256_cvtps_pd(lo3);
            __m256d d7 = _mm256_cvtps_pd(hi3);
            __m256d d8 = _mm256_cvtps_pd(lo4);
            __m256d d9 = _mm256_cvtps_pd(hi4);
            __m256d d10 = _mm256_cvtps_pd(lo5);
            __m256d d11 = _mm256_cvtps_pd(hi5);
            __m256d d12 = _mm256_cvtps_pd(lo6);
            __m256d d13 = _mm256_cvtps_pd(hi6);
            __m256d d14 = _mm256_cvtps_pd(lo7);
            __m256d d15 = _mm256_cvtps_pd(hi7);

            // FMA: acc += d * d
            dacc0 = _mm256_fmadd_pd(d0, d0, dacc0);
            dacc0 = _mm256_fmadd_pd(d1, d1, dacc0);
            dacc0 = _mm256_fmadd_pd(d2, d2, dacc0);
            dacc0 = _mm256_fmadd_pd(d3, d3, dacc0);
            dacc1 = _mm256_fmadd_pd(d4, d4, dacc1);
            dacc1 = _mm256_fmadd_pd(d5, d5, dacc1);
            dacc1 = _mm256_fmadd_pd(d6, d6, dacc1);
            dacc1 = _mm256_fmadd_pd(d7, d7, dacc1);
            dacc2 = _mm256_fmadd_pd(d8, d8, dacc2);
            dacc2 = _mm256_fmadd_pd(d9, d9, dacc2);
            dacc2 = _mm256_fmadd_pd(d10, d10, dacc2);
            dacc2 = _mm256_fmadd_pd(d11, d11, dacc2);
            dacc3 = _mm256_fmadd_pd(d12, d12, dacc3);
            dacc3 = _mm256_fmadd_pd(d13, d13, dacc3);
            dacc3 = _mm256_fmadd_pd(d14, d14, dacc3);
            dacc3 = _mm256_fmadd_pd(d15, d15, dacc3);
        }

        // Horizontal reduction
        __m128d h0 = _mm256_castpd256_pd128(dacc0);
        __m128d h1 = _mm256_extractf128_pd(dacc0, 1);
        __m128d h2 = _mm256_castpd256_pd128(dacc1);
        __m128d h3 = _mm256_extractf128_pd(dacc1, 1);
        __m128d h4 = _mm256_castpd256_pd128(dacc2);
        __m128d h5 = _mm256_extractf128_pd(dacc2, 1);
        __m128d h6 = _mm256_castpd256_pd128(dacc3);
        __m128d h7 = _mm256_extractf128_pd(dacc3, 1);

        __m128d sum01 = _mm_add_pd(h0, h1);
        __m128d sum23 = _mm_add_pd(h2, h3);
        __m128d sum45 = _mm_add_pd(h4, h5);
        __m128d sum67 = _mm_add_pd(h6, h7);

        __m128d sum0123 = _mm_add_pd(sum01, sum23);
        __m128d sum4567 = _mm_add_pd(sum45, sum67);
        __m128d sum_all = _mm_add_pd(sum0123, sum4567);

        __m128d shuf = _mm_shuffle_pd(sum_all, sum_all, 1);
        sum_all = _mm_add_pd(sum_all, shuf);

        double partial[2];
        _mm_storeu_pd(partial, sum_all);
        sum_sq = partial[0];

        // Tail: 8-element blocks
        for (; c + 8 <= cols; c += 8)
        {
            __m256 v = _mm256_loadu_ps(src + c);
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 hi = _mm256_extractf128_ps(v, 1);

            __m256d d0 = _mm256_cvtps_pd(lo);
            __m256d d1 = _mm256_cvtps_pd(hi);

            __m256d sq0 = _mm256_mul_pd(d0, d0);
            __m256d sq1 = _mm256_mul_pd(d1, d1);

            __m128d h0_t = _mm256_castpd256_pd128(sq0);
            __m128d h1_t = _mm256_extractf128_pd(sq0, 1);
            __m128d h2_t = _mm256_castpd256_pd128(sq1);
            __m128d h3_t = _mm256_extractf128_pd(sq1, 1);

            __m128d sum_t = _mm_add_pd(_mm_add_pd(h0_t, h1_t), _mm_add_pd(h2_t, h3_t));
            __m128d shuf_t = _mm_shuffle_pd(sum_t, sum_t, 1);
            sum_t = _mm_add_pd(sum_t, shuf_t);

            double tail_partial[2];
            _mm_storeu_pd(tail_partial, sum_t);
            sum_sq += tail_partial[0];
        }

        // Scalar tail
        for (; c < cols; ++c)
        {
            double val = static_cast<double>(src[c]);
            sum_sq += val * val;
        }

        // Phase 2: Compute inverse RMS
        float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
        float inv_rms = 1.0f / rms;
        if (!std::isfinite(sum_sq))
            rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteSummary,
                "CPU FP32 RMS observed a nonfinite original-row squared summary");
        if (!std::isfinite(inv_rms))
            rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteInverse,
                "CPU FP32 RMS original-row inverse is not representable");
        __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);

        // Phase 3: Apply normalization with AVX2
        c = 0;
        const __m256 ones = _mm256_set1_ps(1.0f);
        for (; c + 8 <= cols; c += 8)
        {
            __m256 src_vec = _mm256_loadu_ps(src + c);
            __m256 gamma_vec = gamma ? _mm256_loadu_ps(gamma + c) : ones;
            __m256 normalized = _mm256_mul_ps(_mm256_mul_ps(src_vec, inv_rms_vec), gamma_vec);
            _mm256_storeu_ps(dst + c, normalized);
        }

        // Scalar tail
        for (; c < cols; ++c)
        {
            dst[c] = src[c] * inv_rms * (gamma ? gamma[c] : 1.0f);
        }
    }
#endif

#if defined(__AVX512F__)
    void rmsnorm_fused_row_avx512(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t cols,
        float epsilon)
    {
        if (cols == 0) return; // Zero-width invocation is a no-op: no values or inverse.
        if (!std::isfinite(epsilon) || epsilon < 0.0f)
            rms_numeric::rejectPolicy("CPU FP32 RMS requires a finite nonnegative original epsilon");
        // Phase 1: Compute sum of squares with AVX512
        double sum_sq = 0.0;
        std::size_t c = 0;

        // Multi-accumulator for better ILP
        __m512d acc0 = _mm512_setzero_pd();
        __m512d acc1 = _mm512_setzero_pd();

        for (; c + 32 <= cols; c += 32)
        {
            // Process 32 float values per iteration
            __m512 fp32_0 = _mm512_loadu_ps(src + c);
            __m512 fp32_1 = _mm512_loadu_ps(src + c + 16);

            // Convert to double and accumulate squares
            __m256 lo0 = _mm512_castps512_ps256(fp32_0);
            __m256 hi0 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(fp32_0), 1));
            __m256 lo1 = _mm512_castps512_ps256(fp32_1);
            __m256 hi1 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(fp32_1), 1));

            __m512d d0a = _mm512_cvtps_pd(lo0);
            __m512d d0b = _mm512_cvtps_pd(hi0);
            __m512d d1a = _mm512_cvtps_pd(lo1);
            __m512d d1b = _mm512_cvtps_pd(hi1);

            // FMA: acc += d * d
            acc0 = _mm512_fmadd_pd(d0a, d0a, acc0);
            acc0 = _mm512_fmadd_pd(d0b, d0b, acc0);
            acc1 = _mm512_fmadd_pd(d1a, d1a, acc1);
            acc1 = _mm512_fmadd_pd(d1b, d1b, acc1);
        }

        sum_sq += _mm512_reduce_add_pd(acc0);
        sum_sq += _mm512_reduce_add_pd(acc1);

        // Scalar tail
        for (; c < cols; ++c)
        {
            float val = src[c];
            sum_sq += static_cast<double>(val) * static_cast<double>(val);
        }

        // Phase 2: Compute inverse RMS
        if (!std::isfinite(sum_sq))
            rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteSummary,
                "CPU FP32 RMS observed a nonfinite original-row squared summary");
        float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
        float inv_rms = 1.0f / rms;
        if (!std::isfinite(inv_rms))
            rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteInverse,
                "CPU FP32 RMS original-row inverse is not representable");
        __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);

        // Phase 3: Normalize and convert back with AVX512
        c = 0;
        const __m512 ones = _mm512_set1_ps(1.0f);
        for (; c + 16 <= cols; c += 16)
        {
            __m512 fp32_vals = _mm512_loadu_ps(src + c);
            __m512 gamma_vec = gamma ? _mm512_loadu_ps(gamma + c) : ones;
            __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(fp32_vals, inv_rms_vec), gamma_vec);
            _mm512_storeu_ps(dst + c, normalized);
        }

        // Scalar tail
        for (; c < cols; ++c)
        {
            float val = src[c];
            float normalized = val * inv_rms * (gamma ? gamma[c] : 1.0f);
            dst[c] = normalized;
        }
    }

    void fused_residual_rmsnorm_row_avx512(
        const float *input,
        float *residual,
        const float *gamma,
        float *output,
        std::size_t cols,
        float epsilon)
    {
        if (cols == 0) return; // Zero-width invocation is a no-op: no residual, output or inverse.
        if (!std::isfinite(epsilon) || epsilon < 0.0f)
            rms_numeric::rejectPolicy("CPU FP32 RMS requires a finite nonnegative original epsilon");
        // Pass 1: residual += input, accumulate sum of squares (double precision)
        double sum_sq = 0.0;
        std::size_t c = 0;

        __m512d acc0 = _mm512_setzero_pd();
        __m512d acc1 = _mm512_setzero_pd();

        for (; c + 32 <= cols; c += 32)
        {
            // First 16 floats
            __m512 v_in0 = _mm512_loadu_ps(input + c);
            __m512 v_res0 = _mm512_loadu_ps(residual + c);
            v_res0 = _mm512_add_ps(v_res0, v_in0);
            _mm512_storeu_ps(residual + c, v_res0);

            __m256 lo0 = _mm512_castps512_ps256(v_res0);
            __m256 hi0 = _mm256_castsi256_ps(
                _mm512_extracti64x4_epi64(_mm512_castps_si512(v_res0), 1));
            acc0 = _mm512_fmadd_pd(_mm512_cvtps_pd(lo0), _mm512_cvtps_pd(lo0), acc0);
            acc0 = _mm512_fmadd_pd(_mm512_cvtps_pd(hi0), _mm512_cvtps_pd(hi0), acc0);

            // Next 16 floats
            __m512 v_in1 = _mm512_loadu_ps(input + c + 16);
            __m512 v_res1 = _mm512_loadu_ps(residual + c + 16);
            v_res1 = _mm512_add_ps(v_res1, v_in1);
            _mm512_storeu_ps(residual + c + 16, v_res1);

            __m256 lo1 = _mm512_castps512_ps256(v_res1);
            __m256 hi1 = _mm256_castsi256_ps(
                _mm512_extracti64x4_epi64(_mm512_castps_si512(v_res1), 1));
            acc1 = _mm512_fmadd_pd(_mm512_cvtps_pd(lo1), _mm512_cvtps_pd(lo1), acc1);
            acc1 = _mm512_fmadd_pd(_mm512_cvtps_pd(hi1), _mm512_cvtps_pd(hi1), acc1);
        }

        sum_sq += _mm512_reduce_add_pd(acc0);
        sum_sq += _mm512_reduce_add_pd(acc1);

        // Scalar tail for pass 1
        for (; c < cols; ++c)
        {
            residual[c] += input[c];
            sum_sq += static_cast<double>(residual[c]) * static_cast<double>(residual[c]);
        }

        // Compute inverse RMS
        if (!std::isfinite(sum_sq))
            rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteSummary,
                "CPU FP32 RMS observed a nonfinite original-row squared summary");
        float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
        float inv_rms = 1.0f / rms;
        if (!std::isfinite(inv_rms))
            rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteInverse,
                "CPU FP32 RMS original-row inverse is not representable");
        __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);

        // Pass 2: output = gamma * residual * inv_rms
        c = 0;
        const __m512 ones = _mm512_set1_ps(1.0f);
        for (; c + 16 <= cols; c += 16)
        {
            __m512 v_res = _mm512_loadu_ps(residual + c);
            __m512 v_gamma = gamma ? _mm512_loadu_ps(gamma + c) : ones;
            __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(v_res, inv_rms_vec), v_gamma);
            _mm512_storeu_ps(output + c, normalized);
        }

        // Scalar tail for pass 2
        for (; c < cols; ++c)
        {
            output[c] = residual[c] * inv_rms * (gamma ? gamma[c] : 1.0f);
        }
    }
#endif

    // ========================================================================
    // BF16 RMSNorm Implementation (Scalar/AVX2/AVX512)
    // ========================================================================

    // ========== BF16 Conversion Helpers (Moved to top) ==========

    // (Moved to top)

    namespace detail
    {

#if defined(__aarch64__) && defined(__ARM_NEON)
        /** @brief Normalize one original encoded row with native arithmetic and exact storage tails.
         * @tparam Encoding Original FP16/BF16 storage kind.
         * @param src Original row; @param gamma Immutable original weights, or identity when null.
         * @param dst Original writable row; @param cols Logical count; @param epsilon Original policy.
         */
        template<neon_rmsnorm::Storage Encoding>
        void native_encoded_rmsnorm_row(const uint16_t *src, const float *gamma,
                                        uint16_t *dst, std::size_t cols, float epsilon)
        {
            using namespace neon_rmsnorm;
            normalize(OriginalRows<Encoding, 1>::bind({std::span<const uint16_t>(src, cols)}, cols),
                std::array{std::span<uint16_t>(dst, cols)},
                gamma ? std::span<const float>(gamma, cols) : std::span<const float>{}, epsilon);
        }
        /** @brief Native BF16 original-row normalization with portable RNE publication.
         * @param src Original BF16; @param gamma Original FP32 gamma; @param dst Original output.
         * @param cols Logical elements; @param epsilon Original normalization policy.
         */
        void bf16_rmsnorm_row_neon(const uint16_t *src, const float *gamma,
                                   uint16_t *dst, std::size_t cols, float epsilon)
        {
            native_encoded_rmsnorm_row<neon_rmsnorm::Storage::BF16>(src, gamma, dst, cols, epsilon);
        }
        /** @brief Native FP16 original-row normalization with the FP16Utils storage program.
         * @param src Original FP16; @param gamma Original FP32 gamma; @param dst Original output.
         * @param cols Logical elements; @param epsilon Original normalization policy.
         */
        void fp16_rmsnorm_row_neon(const uint16_t *src, const float *gamma,
                                   uint16_t *dst, std::size_t cols, float epsilon)
        {
            native_encoded_rmsnorm_row<neon_rmsnorm::Storage::FP16>(src, gamma, dst, cols, epsilon);
        }
#endif
        // ========== BF16 RMSNorm Kernels ==========

        /// BF16 RMSNorm row kernel - Scalar path
        void bf16_rmsnorm_row_scalar(
            const uint16_t *src_row,
            const float *gamma,
            uint16_t *dst_row,
            std::size_t cols,
            float epsilon)
        {
            if (cols == 0) return; // Zero-width invocation is a no-op: no values or inverse.
            if (!std::isfinite(epsilon) || epsilon < 0.0f)
                rms_numeric::rejectPolicy("CPU BF16 RMS requires a finite nonnegative original epsilon");
            // Phase 1: Compute sum of squares
            double sum_sq = 0.0;
            for (std::size_t c = 0; c < cols; ++c)
            {
                float val = bf16_to_fp32_scalar(src_row[c]);
                sum_sq += static_cast<double>(val) * static_cast<double>(val);
            }
            if (!std::isfinite(sum_sq))
                rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteSummary,
                    "CPU BF16 RMS observed a nonfinite original-row squared summary");

            // Phase 2: Compute inverse RMS (original FP32-mean program)
            const float denominator = float(sum_sq / double(cols)) + epsilon;
            const float inv_rms = 1.0f / std::sqrt(denominator);
            if (!std::isfinite(inv_rms))
                rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteInverse,
                    "CPU BF16 RMS original-row inverse is not representable");

            // Phase 3: Normalize and convert back; null gamma is admitted identity.
            if (gamma)
            {
                for (std::size_t c = 0; c < cols; ++c)
                {
                    float val = bf16_to_fp32_scalar(src_row[c]);
                    float normalized = val * inv_rms * gamma[c];
                    dst_row[c] = fp32_to_bf16_scalar(normalized);
                }
            }
            else
            {
                for (std::size_t c = 0; c < cols; ++c)
                {
                    float val = bf16_to_fp32_scalar(src_row[c]);
                    float normalized = val * inv_rms;
                    dst_row[c] = fp32_to_bf16_scalar(normalized);
                }
            }
        }

#if defined(__AVX512F__)
        /// BF16 RMSNorm row kernel - AVX512 path
        void bf16_rmsnorm_row_avx512(
            const uint16_t *src_row,
            const float *gamma,
            uint16_t *dst_row,
            std::size_t cols,
            float epsilon)
        {
            if (cols == 0) return; // Zero-width invocation is a no-op: no values or inverse.
            if (!std::isfinite(epsilon) || epsilon < 0.0f)
                rms_numeric::rejectPolicy("CPU BF16 RMS requires a finite nonnegative original epsilon");
            // Phase 1: Compute sum of squares with AVX512
            double sum_sq = 0.0;
            std::size_t c = 0;

            __m512d acc0 = _mm512_setzero_pd();
            __m512d acc1 = _mm512_setzero_pd();

            for (; c + 32 <= cols; c += 32)
            {
                // Process 32 BF16 values per iteration
                __m512 fp32_0 = bf16_to_fp32_avx512(src_row + c);
                __m512 fp32_1 = bf16_to_fp32_avx512(src_row + c + 16);

                // Convert to double and accumulate squares
                __m256 lo0 = _mm512_castps512_ps256(fp32_0);
                __m256 hi0 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(fp32_0), 1));
                __m256 lo1 = _mm512_castps512_ps256(fp32_1);
                __m256 hi1 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(fp32_1), 1));

                __m512d d0a = _mm512_cvtps_pd(lo0);
                __m512d d0b = _mm512_cvtps_pd(hi0);
                __m512d d1a = _mm512_cvtps_pd(lo1);
                __m512d d1b = _mm512_cvtps_pd(hi1);

                acc0 = _mm512_fmadd_pd(d0a, d0a, acc0);
                acc0 = _mm512_fmadd_pd(d0b, d0b, acc0);
                acc1 = _mm512_fmadd_pd(d1a, d1a, acc1);
                acc1 = _mm512_fmadd_pd(d1b, d1b, acc1);
            }

            sum_sq += _mm512_reduce_add_pd(acc0);
            sum_sq += _mm512_reduce_add_pd(acc1);

            // Scalar tail
            for (; c < cols; ++c)
            {
                float val = bf16_to_fp32_scalar(src_row[c]);
                sum_sq += static_cast<double>(val) * static_cast<double>(val);
            }
            if (!std::isfinite(sum_sq))
                rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteSummary,
                    "CPU BF16 RMS observed a nonfinite original-row squared summary");

            // Phase 2: Compute inverse RMS
            float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
            float inv_rms = 1.0f / rms;
            if (!std::isfinite(inv_rms))
                rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteInverse,
                    "CPU BF16 RMS original-row inverse is not representable");
            __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);

            // Phase 3: Normalize and convert back with AVX512
            c = 0;
            const __m512 ones = _mm512_set1_ps(1.0f);
            for (; c + 16 <= cols; c += 16)
            {
                __m512 fp32_vals = bf16_to_fp32_avx512(src_row + c);
                __m512 gamma_vec = gamma ? _mm512_loadu_ps(gamma + c) : ones;
                __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(fp32_vals, inv_rms_vec), gamma_vec);
                fp32_to_bf16_avx512(normalized, dst_row + c);
            }

            // Scalar tail
            for (; c < cols; ++c)
            {
                float val = bf16_to_fp32_scalar(src_row[c]);
                float normalized = val * inv_rms * (gamma ? gamma[c] : 1.0f);
                dst_row[c] = fp32_to_bf16_scalar(normalized);
            }
        }
#endif

        // ========== FP16 RMSNorm Kernels ==========

        /// FP16 RMSNorm row kernel - Scalar path
        void fp16_rmsnorm_row_scalar(
            const uint16_t *src_row,
            const float *gamma,
            uint16_t *dst_row,
            std::size_t cols,
            float epsilon)
        {
            if (cols == 0) return; // Zero-width invocation is a no-op: no values or inverse.
            if (!std::isfinite(epsilon) || epsilon < 0.0f)
                rms_numeric::rejectPolicy("CPU FP16 RMS requires a finite nonnegative original epsilon");
            // Phase 1: Compute sum of squares
            double sum_sq = 0.0;
            for (std::size_t c = 0; c < cols; ++c)
            {
                float val = fp16_to_fp32_scalar(src_row[c]);
                sum_sq += static_cast<double>(val) * static_cast<double>(val);
            }
            if (!std::isfinite(sum_sq))
                rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteSummary,
                    "CPU FP16 RMS observed a nonfinite original-row squared summary");

            // Phase 2: Compute inverse RMS (original FP32-mean program)
            const float denominator = float(sum_sq / double(cols)) + epsilon;
            const float inv_rms = 1.0f / std::sqrt(denominator);
            if (!std::isfinite(inv_rms))
                rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteInverse,
                    "CPU FP16 RMS original-row inverse is not representable");

            // Phase 3: Normalize and convert back; null gamma is admitted identity.
            if (gamma)
            {
                for (std::size_t c = 0; c < cols; ++c)
                {
                    float val = fp16_to_fp32_scalar(src_row[c]);
                    float normalized = val * inv_rms * gamma[c];
                    dst_row[c] = fp32_to_fp16_scalar(normalized);
                }
            }
            else
            {
                for (std::size_t c = 0; c < cols; ++c)
                {
                    float val = fp16_to_fp32_scalar(src_row[c]);
                    float normalized = val * inv_rms;
                    dst_row[c] = fp32_to_fp16_scalar(normalized);
                }
            }
        }

#if defined(__AVX512F__)
        /// FP16 RMSNorm row kernel - AVX512 path
        void fp16_rmsnorm_row_avx512(
            const uint16_t *src_row,
            const float *gamma,
            uint16_t *dst_row,
            std::size_t cols,
            float epsilon)
        {
            if (cols == 0) return; // Zero-width invocation is a no-op: no values or inverse.
            if (!std::isfinite(epsilon) || epsilon < 0.0f)
                rms_numeric::rejectPolicy("CPU FP16 RMS requires a finite nonnegative original epsilon");
            // Phase 1: Compute sum of squares with AVX512
            double sum_sq = 0.0;
            std::size_t c = 0;

            __m512d acc0 = _mm512_setzero_pd();
            __m512d acc1 = _mm512_setzero_pd();

            for (; c + 32 <= cols; c += 32)
            {
                // Process 32 FP16 values per iteration
                __m512 fp32_0 = fp16_to_fp32_avx512(src_row + c);
                __m512 fp32_1 = fp16_to_fp32_avx512(src_row + c + 16);

                // Convert to double and accumulate squares
                __m256 lo0 = _mm512_castps512_ps256(fp32_0);
                __m256 hi0 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(fp32_0), 1));
                __m256 lo1 = _mm512_castps512_ps256(fp32_1);
                __m256 hi1 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(fp32_1), 1));

                __m512d d0a = _mm512_cvtps_pd(lo0);
                __m512d d0b = _mm512_cvtps_pd(hi0);
                __m512d d1a = _mm512_cvtps_pd(lo1);
                __m512d d1b = _mm512_cvtps_pd(hi1);

                acc0 = _mm512_fmadd_pd(d0a, d0a, acc0);
                acc0 = _mm512_fmadd_pd(d0b, d0b, acc0);
                acc1 = _mm512_fmadd_pd(d1a, d1a, acc1);
                acc1 = _mm512_fmadd_pd(d1b, d1b, acc1);
            }

            sum_sq += _mm512_reduce_add_pd(acc0);
            sum_sq += _mm512_reduce_add_pd(acc1);

            // Scalar tail
            for (; c < cols; ++c)
            {
                float val = fp16_to_fp32_scalar(src_row[c]);
                sum_sq += static_cast<double>(val) * static_cast<double>(val);
            }
            if (!std::isfinite(sum_sq))
                rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteSummary,
                    "CPU FP16 RMS observed a nonfinite original-row squared summary");

            // Phase 2: Compute inverse RMS
            float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
            float inv_rms = 1.0f / rms;
            if (!std::isfinite(inv_rms))
                rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteInverse,
                    "CPU FP16 RMS original-row inverse is not representable");
            __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);

            // Phase 3: Normalize and convert back with AVX512
            c = 0;
            const __m512 ones = _mm512_set1_ps(1.0f);
            for (; c + 16 <= cols; c += 16)
            {
                __m512 fp32_vals = fp16_to_fp32_avx512(src_row + c);
                __m512 gamma_vec = gamma ? _mm512_loadu_ps(gamma + c) : ones;
                __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(fp32_vals, inv_rms_vec), gamma_vec);
                fp32_to_fp16_avx512(normalized, dst_row + c);
            }

            // Scalar tail
            for (; c < cols; ++c)
            {
                float val = fp16_to_fp32_scalar(src_row[c]);
                float normalized = val * inv_rms * (gamma ? gamma[c] : 1.0f);
                dst_row[c] = fp32_to_fp16_scalar(normalized);
            }
        }
#endif

#if defined(__AVX2__) && defined(__F16C__)
        /// FP16 RMSNorm row kernel - AVX2+F16C path
        void fp16_rmsnorm_row_avx2(
            const uint16_t *src_row,
            const float *gamma,
            uint16_t *dst_row,
            std::size_t cols,
            float epsilon)
        {
            if (cols == 0) return; // Zero-width invocation is a no-op: no values or inverse.
            if (!std::isfinite(epsilon) || epsilon < 0.0f)
                rms_numeric::rejectPolicy("CPU FP16 RMS requires a finite nonnegative original epsilon");
            // Phase 1: Compute sum of squares with AVX2
            float sum_sq_f = 0.0f;
            std::size_t c = 0;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            for (; c + 16 <= cols; c += 16)
            {
                __m256 fp32_0 = fp16_to_fp32_avx2(src_row + c);
                __m256 fp32_1 = fp16_to_fp32_avx2(src_row + c + 8);

                acc0 = _mm256_fmadd_ps(fp32_0, fp32_0, acc0);
                acc1 = _mm256_fmadd_ps(fp32_1, fp32_1, acc1);
            }

            // Horizontal reduction
            __m256 acc_combined = _mm256_add_ps(acc0, acc1);
            __m128 lo = _mm256_castps256_ps128(acc_combined);
            __m128 hi = _mm256_extractf128_ps(acc_combined, 1);
            __m128 sum4 = _mm_add_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(sum4);
            __m128 sums = _mm_add_ps(sum4, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            _mm_store_ss(&sum_sq_f, sums);

            // Scalar tail
            for (; c < cols; ++c)
            {
                float val = fp16_to_fp32_scalar(src_row[c]);
                sum_sq_f += val * val;
            }

            if (!std::isfinite(sum_sq_f))
                rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteSummary,
                    "CPU FP16 RMS observed a nonfinite original-row squared summary");

            // Phase 2: Compute inverse RMS
            float rms = std::sqrt(sum_sq_f / cols + epsilon);
            float inv_rms = 1.0f / rms;
            if (!std::isfinite(inv_rms))
                rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteInverse,
                    "CPU FP16 RMS original-row inverse is not representable");
            __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);

            // Phase 3: Normalize and convert back
            c = 0;
            const __m256 ones = _mm256_set1_ps(1.0f);
            for (; c + 8 <= cols; c += 8)
            {
                __m256 fp32_vals = fp16_to_fp32_avx2(src_row + c);
                __m256 gamma_vec = gamma ? _mm256_loadu_ps(gamma + c) : ones;
                __m256 normalized = _mm256_mul_ps(_mm256_mul_ps(fp32_vals, inv_rms_vec), gamma_vec);
                fp32_to_fp16_avx2(normalized, dst_row + c);
            }

            // Scalar tail
            for (; c < cols; ++c)
            {
                float val = fp16_to_fp32_scalar(src_row[c]);
                float normalized = val * inv_rms * (gamma ? gamma[c] : 1.0f);
                dst_row[c] = fp32_to_fp16_scalar(normalized);
            }
        }
#endif

        // --- ISA stubs for runtime dispatch (bf16/fp16 rmsnorm) ---
        void bf16_rmsnorm_row_avx2(
            const uint16_t *src_row, const float *gamma,
            uint16_t *dst_row, std::size_t cols, float epsilon)
        {
            bf16_rmsnorm_row_scalar(src_row, gamma, dst_row, cols, epsilon);
        }
#if !defined(__AVX512F__)
        void bf16_rmsnorm_row_avx512(
            const uint16_t *src_row, const float *gamma,
            uint16_t *dst_row, std::size_t cols, float epsilon)
        {
            bf16_rmsnorm_row_scalar(src_row, gamma, dst_row, cols, epsilon);
        }
#endif
#if !(defined(__AVX2__) && defined(__F16C__))
        void fp16_rmsnorm_row_avx2(
            const uint16_t *src_row, const float *gamma,
            uint16_t *dst_row, std::size_t cols, float epsilon)
        {
            fp16_rmsnorm_row_scalar(src_row, gamma, dst_row, cols, epsilon);
        }
#endif
#if !defined(__AVX512F__)
        void fp16_rmsnorm_row_avx512(
            const uint16_t *src_row, const float *gamma,
            uint16_t *dst_row, std::size_t cols, float epsilon)
        {
            fp16_rmsnorm_row_avx2(src_row, gamma, dst_row, cols, epsilon);
        }
#endif

    } // namespace detail
    using namespace detail;

    // ========================================================================
    // Public Implementations
    // ========================================================================

    void rmsnorm_fused_bf16_vectorized(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts)
    {
        const auto bank=rmsnorm_operands::FloatingBorrow<uint16_t>::bind(src,gamma,dst,rows,cols,epsilon);
        bank.requireMetadataReadOnly(&opts,sizeof(opts));
        // BF16 has no declared T5 FP32 accumulation program; reject cold before
        // any workshare so no worker sees an exception it cannot propagate.
        if (opts.t5_compat_mode)
            rms_numeric::rejectPolicy("BF16 RMS has no declared T5 FP32 accumulation program");
        if (!::llaminar::v2::beginWorkshare(!bank.empty())) return;
#if defined(__aarch64__) && defined(__ARM_NEON)
        if (use_native_rmsnorm(opts))
        {
            native_floating_matrix<neon_rmsnorm::Storage::BF16>(bank,opts);
            return;
        }
#endif
        auto work=[&]() {
#pragma omp for schedule(static)
            for(std::ptrdiff_t row=0;row<std::ptrdiff_t(rows);++row)
            {
                const auto input=bank.sourceRow(std::size_t(row));
                const auto output=bank.outputRow(std::size_t(row));
                if(opts.force_scalar) bf16_rmsnorm_row_scalar(input.data(),gamma,output.data(),cols,epsilon);
                else ISA_DISPATCH_VOID(bf16_rmsnorm_row,input.data(),gamma,output.data(),cols,epsilon);
            }
        };
        OMP_WORKSHARE_REGION_IF(work,omp_in_parallel() || (rows>1 && want_parallel(rows,cols,opts)));
    }

    void rmsnorm_fused_fp16_vectorized(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts)
    {
        const auto bank=rmsnorm_operands::FloatingBorrow<uint16_t>::bind(src,gamma,dst,rows,cols,epsilon);
        bank.requireMetadataReadOnly(&opts,sizeof(opts));
        // FP16 has no declared T5 FP32 accumulation program; reject cold before
        // any workshare so no worker sees an exception it cannot propagate.
        if (opts.t5_compat_mode)
            rms_numeric::rejectPolicy("FP16 RMS has no declared T5 FP32 accumulation program");
        if (!::llaminar::v2::beginWorkshare(!bank.empty())) return;
#if defined(__aarch64__) && defined(__ARM_NEON)
        if (use_native_rmsnorm(opts))
        {
            native_floating_matrix<neon_rmsnorm::Storage::FP16>(bank,opts);
            return;
        }
#endif
        auto work=[&]() {
#pragma omp for schedule(static)
            for(std::ptrdiff_t row=0;row<std::ptrdiff_t(rows);++row)
            {
                const auto input=bank.sourceRow(std::size_t(row));
                const auto output=bank.outputRow(std::size_t(row));
                if(opts.force_scalar) fp16_rmsnorm_row_scalar(input.data(),gamma,output.data(),cols,epsilon);
                else ISA_DISPATCH_VOID(fp16_rmsnorm_row,input.data(),gamma,output.data(),cols,epsilon);
            }
        };
        OMP_WORKSHARE_REGION_IF(work,omp_in_parallel() || (rows>1 && want_parallel(rows,cols,opts)));
    }

    void rmsnorm_quantize_row_scalar(
        const float *input_row,
        const float *weight,
        int8_t *output_row,
        float &out_scale,
        int cols,
        float epsilon)
    {
        // 1. Compute RMS
        double sum_sq = 0.0;
        for (int i = 0; i < cols; ++i)
        {
            sum_sq += input_row[i] * input_row[i];
        }
        float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
        float inv_rms = 1.0f / rms;

        // 2. Find max abs of normalized values
        float max_abs = 0.0f;
        for (int i = 0; i < cols; ++i)
        {
            float val = input_row[i] * inv_rms * weight[i];
            max_abs = std::max(max_abs, std::abs(val));
        }

        // 3. Compute scale and quantize
        float scale = max_abs / 127.0f;
        if (scale < 1e-8f)
            scale = 1e-8f;
        out_scale = scale;
        float inv_scale = 1.0f / scale;

        for (int i = 0; i < cols; ++i)
        {
            float val = input_row[i] * inv_rms * weight[i];
            int32_t q = std::round(val * inv_scale);
            if (q > 127)
                q = 127;
            if (q < -127)
                q = -127;
            output_row[i] = static_cast<int8_t>(q);
        }
    }

    void rmsnorm_quantize_row_avx2(
        const float *input_row,
        const float *weight,
        int8_t *output_row,
        float &out_scale,
        int cols,
        float epsilon)
    {
#if defined(__AVX2__)
        // 1. Compute RMS
        __m256 sum_sq_vec = _mm256_setzero_ps();
        int i = 0;
        for (; i <= cols - 8; i += 8)
        {
            __m256 v = _mm256_loadu_ps(input_row + i);
            sum_sq_vec = _mm256_fmadd_ps(v, v, sum_sq_vec);
        }
        // Reduce sum_sq_vec
        __m128 lo = _mm256_castps256_ps128(sum_sq_vec);
        __m128 hi = _mm256_extractf128_ps(sum_sq_vec, 1);
        __m128 sum4 = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(sum4);
        __m128 sums = _mm_add_ps(sum4, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        float sum_sq = _mm_cvtss_f32(sums);

        for (; i < cols; ++i)
        {
            sum_sq += input_row[i] * input_row[i];
        }

        float rms = std::sqrt(sum_sq / cols + epsilon);
        float inv_rms = 1.0f / rms;
        __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);

        // 2. Find max abs
        __m256 max_abs_vec = _mm256_setzero_ps();
        __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

        i = 0;
        for (; i <= cols - 8; i += 8)
        {
            __m256 v = _mm256_loadu_ps(input_row + i);
            __m256 w = _mm256_loadu_ps(weight + i);
            __m256 norm = _mm256_mul_ps(_mm256_mul_ps(v, inv_rms_vec), w);
            __m256 abs_norm = _mm256_and_ps(norm, abs_mask);
            max_abs_vec = _mm256_max_ps(max_abs_vec, abs_norm);
        }
        // Reduce max_abs_vec
        lo = _mm256_castps256_ps128(max_abs_vec);
        hi = _mm256_extractf128_ps(max_abs_vec, 1);
        __m128 max4 = _mm_max_ps(lo, hi);
        shuf = _mm_movehdup_ps(max4);
        __m128 maxs = _mm_max_ps(max4, shuf);
        shuf = _mm_movehl_ps(shuf, maxs);
        maxs = _mm_max_ss(maxs, shuf);
        float max_abs = _mm_cvtss_f32(maxs);

        for (; i < cols; ++i)
        {
            float val = input_row[i] * inv_rms * weight[i];
            max_abs = std::max(max_abs, std::abs(val));
        }

        // 3. Compute scale and quantize
        float scale = max_abs / 127.0f;
        if (scale < 1e-8f)
            scale = 1e-8f;
        out_scale = scale;
        float inv_scale = 1.0f / scale;
        __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);

        i = 0;
        for (; i <= cols - 8; i += 8)
        {
            __m256 v = _mm256_loadu_ps(input_row + i);
            __m256 w = _mm256_loadu_ps(weight + i);
            __m256 norm = _mm256_mul_ps(_mm256_mul_ps(v, inv_rms_vec), w);
            __m256 scaled = _mm256_mul_ps(norm, inv_scale_vec);
            __m256i q = _mm256_cvtps_epi32(scaled); // Round to nearest integer

            int32_t tmp[8];
            _mm256_storeu_si256((__m256i *)tmp, q);
            for (int k = 0; k < 8; ++k)
            {
                int32_t val = tmp[k];
                if (val > 127)
                    val = 127;
                if (val < -127)
                    val = -127;
                output_row[i + k] = (int8_t)val;
            }
        }

        for (; i < cols; ++i)
        {
            float val = input_row[i] * inv_rms * weight[i];
            int32_t q = std::round(val * inv_scale);
            if (q > 127)
                q = 127;
            if (q < -127)
                q = -127;
            output_row[i] = static_cast<int8_t>(q);
        }
#else
        rmsnorm_quantize_row_scalar(input_row, weight, output_row, out_scale, cols, epsilon);
#endif
    }

    void rmsnorm_quantize_row_avx512(
        const float *input_row,
        const float *weight,
        int8_t *output_row,
        float &out_scale,
        int cols,
        float epsilon)
    {
#if defined(__AVX512F__)
        // 1. Compute RMS
        __m512 sum_sq_vec = _mm512_setzero_ps();
        int i = 0;
        for (; i <= cols - 16; i += 16)
        {
            __m512 v = _mm512_loadu_ps(input_row + i);
            sum_sq_vec = _mm512_fmadd_ps(v, v, sum_sq_vec);
        }
        float sum_sq = _mm512_reduce_add_ps(sum_sq_vec);

        for (; i < cols; ++i)
        {
            sum_sq += input_row[i] * input_row[i];
        }

        float rms = std::sqrt(sum_sq / cols + epsilon);
        float inv_rms = 1.0f / rms;
        __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);

        // 2. Find max abs
        __m512 max_abs_vec = _mm512_setzero_ps();
        __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));

        i = 0;
        for (; i <= cols - 16; i += 16)
        {
            __m512 v = _mm512_loadu_ps(input_row + i);
            __m512 w = _mm512_loadu_ps(weight + i);
            __m512 norm = _mm512_mul_ps(_mm512_mul_ps(v, inv_rms_vec), w);
            __m512 abs_norm = _mm512_and_ps(norm, abs_mask);
            max_abs_vec = _mm512_max_ps(max_abs_vec, abs_norm);
        }
        float max_abs = _mm512_reduce_max_ps(max_abs_vec);

        for (; i < cols; ++i)
        {
            float val = input_row[i] * inv_rms * weight[i];
            max_abs = std::max(max_abs, std::abs(val));
        }

        // 3. Compute scale and quantize
        float scale = max_abs / 127.0f;
        if (scale < 1e-8f)
            scale = 1e-8f;
        out_scale = scale;
        float inv_scale = 1.0f / scale;
        __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);

        i = 0;
        for (; i <= cols - 16; i += 16)
        {
            __m512 v = _mm512_loadu_ps(input_row + i);
            __m512 w = _mm512_loadu_ps(weight + i);
            __m512 norm = _mm512_mul_ps(_mm512_mul_ps(v, inv_rms_vec), w);
            __m512 scaled = _mm512_mul_ps(norm, inv_scale_vec);
            __m512i q = _mm512_cvtps_epi32(scaled);

#if defined(__AVX512BW__)
            __m128i q8 = _mm512_cvtepi32_epi8(q);
            _mm_storeu_si128((__m128i *)(output_row + i), q8);
#else
            int32_t tmp[16];
            _mm512_storeu_si512(tmp, q);
            for (int k = 0; k < 16; ++k)
            {
                int32_t val = tmp[k];
                if (val > 127)
                    val = 127;
                if (val < -127)
                    val = -127;
                output_row[i + k] = (int8_t)val;
            }
#endif
        }

        for (; i < cols; ++i)
        {
            float val = input_row[i] * inv_rms * weight[i];
            int32_t q = std::round(val * inv_scale);
            if (q > 127)
                q = 127;
            if (q < -127)
                q = -127;
            output_row[i] = static_cast<int8_t>(q);
        }
#else
        rmsnorm_quantize_row_scalar(input_row, weight, output_row, out_scale, cols, epsilon);
#endif
    }

    // Implement per-row primitives (wrappers)
    void rmsnorm_fused_row_bf16_scalar(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon)
    {
        bf16_rmsnorm_row_scalar(src, gamma, dst, cols, epsilon);
    }

    void rmsnorm_fused_row_bf16_avx2(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon)
    {
        // Fallback to scalar if no AVX2 BF16 implementation
        bf16_rmsnorm_row_scalar(src, gamma, dst, cols, epsilon);
    }

    void rmsnorm_fused_row_bf16_avx512(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon)
    {
#if defined(__AVX512F__)
        bf16_rmsnorm_row_avx512(src, gamma, dst, cols, epsilon);
#else
        bf16_rmsnorm_row_scalar(src, gamma, dst, cols, epsilon);
#endif
    }

    void rmsnorm_fused_row_fp16_scalar(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon)
    {
        fp16_rmsnorm_row_scalar(src, gamma, dst, cols, epsilon);
    }

    void rmsnorm_fused_row_fp16_avx2(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon)
    {
#if defined(__AVX2__) && defined(__F16C__)
        fp16_rmsnorm_row_avx2(src, gamma, dst, cols, epsilon);
#else
        fp16_rmsnorm_row_scalar(src, gamma, dst, cols, epsilon);
#endif
    }

    void rmsnorm_fused_row_fp16_avx512(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon)
    {
#if defined(__AVX512F__)
        fp16_rmsnorm_row_avx512(src, gamma, dst, cols, epsilon);
#else
        fp16_rmsnorm_row_scalar(src, gamma, dst, cols, epsilon);
#endif
    }

    // ========================================================================
    // Q8_1 Integer-Space RMSNorm Implementation
    // ========================================================================

    namespace detail
    {
        /**
         * @brief FP16 to FP32 conversion (for Q8_1 scale factors)
         */
        __attribute__((always_inline)) inline float fp16_to_fp32_q8(uint16_t h)
        {
            // Use hardware instruction if available
#if defined(__F16C__)
            return _cvtsh_ss(h);
#else
            // Software fallback
            uint32_t sign = (h & 0x8000U) << 16;
            uint32_t exp = (h & 0x7C00U) >> 10;
            uint32_t mantissa = h & 0x03FFU;
            uint32_t fp32_bits;
            if (exp == 0)
            {
                if (mantissa == 0)
                    fp32_bits = sign;
                else
                {
                    exp = 1;
                    while ((mantissa & 0x0400U) == 0)
                    {
                        mantissa <<= 1;
                        exp--;
                    }
                    mantissa &= 0x03FFU;
                    fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
                }
            }
            else if (exp == 0x1F)
            {
                fp32_bits = sign | 0x7F800000U | (mantissa << 13);
            }
            else
            {
                fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
            }
            float result;
            std::memcpy(&result, &fp32_bits, sizeof(float));
            return result;
#endif
        }

        /**
         * @brief FP32 to FP16 conversion (for Q8_1 scale factors)
         */
        __attribute__((always_inline)) inline uint16_t fp32_to_fp16_q8(float f)
        {
#if defined(__F16C__)
            return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
#else
            uint32_t bits;
            std::memcpy(&bits, &f, sizeof(float));
            uint32_t sign = (bits >> 16) & 0x8000U;
            int32_t exp = ((bits >> 23) & 0xFFU) - 127 + 15;
            uint32_t mantissa = (bits >> 13) & 0x03FFU;
            if (exp <= 0)
            {
                if (exp < -10)
                    return sign;
                mantissa |= 0x0400U;
                mantissa >>= (1 - exp);
                return sign | mantissa;
            }
            else if (exp >= 0x1F)
            {
                return sign | 0x7C00U;
            }
            else
            {
                return sign | (exp << 10) | mantissa;
            }
#endif
        }

#if defined(__AVX512F__)
        /**
         * @brief Compute sum of squares for 32 int8 values using AVX512
         *
         * Uses VPMADDUBSW + VPMADDWD for efficient int8*int8 -> int32 accumulation.
         * Note: We treat qs as signed int8, so we use a different approach than VNNI.
         */
        int32_t compute_int8_sumsq_avx512(const int8_t *qs)
        {
            // Load 32 int8 values
            __m256i v8 = _mm256_loadu_si256((const __m256i *)qs);

            // Convert to int16 (need signed extension)
            __m512i v16 = _mm512_cvtepi8_epi16(v8);

            // Square: v16 * v16 -> int32 (using PMADDWD with self)
            // PMADDWD: (a0*b0 + a1*b1), (a2*b2 + a3*b3), ...
            __m512i v32 = _mm512_madd_epi16(v16, v16);

            // Horizontal sum
            return _mm512_reduce_add_epi32(v32);
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief Compute sum of squares for 32 int8 values using AVX2
         */
        int32_t compute_int8_sumsq_avx2(const int8_t *qs)
        {
            // Load 32 int8 values as two 128-bit chunks
            __m128i v8_lo = _mm_loadu_si128((const __m128i *)qs);
            __m128i v8_hi = _mm_loadu_si128((const __m128i *)(qs + 16));

            // Convert to int16
            __m256i v16_lo = _mm256_cvtepi8_epi16(v8_lo);
            __m256i v16_hi = _mm256_cvtepi8_epi16(v8_hi);

            // Square using PMADDWD
            __m256i v32_lo = _mm256_madd_epi16(v16_lo, v16_lo);
            __m256i v32_hi = _mm256_madd_epi16(v16_hi, v16_hi);

            // Add the two halves
            __m256i v32_sum = _mm256_add_epi32(v32_lo, v32_hi);

            // Horizontal sum (AVX2 doesn't have _mm256_reduce_add_epi32)
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(v32_sum),
                                           _mm256_extracti128_si256(v32_sum, 1));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
            return _mm_cvtsi128_si32(sum128);
        }
#endif

        /**
         * @brief Compute sum of squares for 32 int8 values (scalar fallback)
         */
        int32_t compute_int8_sumsq_scalar(const int8_t *qs)
        {
            int32_t sum = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t val = static_cast<int32_t>(qs[i]);
                sum += val * val;
            }
            return sum;
        }

        /**
         * @brief Dispatch to best available SIMD for int8 sum-of-squares
         */
        __attribute__((always_inline)) inline int32_t compute_int8_sumsq(const int8_t *qs)
        {
#if defined(__AVX512F__)
            return compute_int8_sumsq_avx512(qs);
#elif defined(__AVX2__)
            return compute_int8_sumsq_avx2(qs);
#else
            return compute_int8_sumsq_scalar(qs);
#endif
        }

    } // namespace detail
    using namespace detail;

    // =========================================================================
    // PURE INTEGER RMSNorm Implementation
    // =========================================================================
    //
    // This implementation performs RMSNorm entirely in integer domain by:
    // 1. Computing sum(qs²) in pure integer (no FP16 scale involved yet)
    // 2. Using a simplified model: assume all blocks have similar scale
    //    OR use a "representative scale" approach
    // 3. Integer inverse square root via Newton-Raphson
    // 4. Pre-quantized gamma weights (int16)
    //
    // KEY SIMPLIFICATION: For activations, blocks within a row typically have
    // similar scales (within 2-4x). We use the geometric mean of scales.
    // =========================================================================

    namespace
    {
#if defined(__AVX512F__)
        /**
         * @brief AVX512 vectorized sum of squares for 32 int8 values
         */
        __attribute__((always_inline)) inline int32_t compute_sumsq_avx512(const int8_t *qs)
        {
            __m256i v8 = _mm256_loadu_si256((const __m256i *)qs);
            __m512i v16 = _mm512_cvtepi8_epi16(v8);
            __m512i v32 = _mm512_madd_epi16(v16, v16);
            return _mm512_reduce_add_epi32(v32);
        }
#endif

#if defined(__AVX2__)
        __attribute__((always_inline)) inline int32_t reduce_add_i32_avx2(__m256i v)
        {
            __m128i lo = _mm256_castsi256_si128(v);
            __m128i hi = _mm256_extracti128_si256(v, 1);
            __m128i sum = _mm_add_epi32(lo, hi);
            sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
            sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
            return _mm_cvtsi128_si32(sum);
        }

        __attribute__((always_inline)) inline int32_t reduce_max_i32_avx2(__m256i v)
        {
            __m128i lo = _mm256_castsi256_si128(v);
            __m128i hi = _mm256_extracti128_si256(v, 1);
            __m128i m = _mm_max_epi32(lo, hi);
            m = _mm_max_epi32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(1, 0, 3, 2)));
            m = _mm_max_epi32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(2, 3, 0, 1)));
            return _mm_cvtsi128_si32(m);
        }

        __attribute__((always_inline)) inline void rmsnorm_q8_1_apply_block_float_scaled_avx2(
            const Q8_1Block &in_blk,
            const int16_t *gamma_block,
            Q8_1Block &out_blk,
            float inv_rms_float)
        {
            alignas(32) int32_t prod[32];
            __m256i max_abs_v = _mm256_setzero_si256();

            for (int i = 0; i < 32; i += 8)
            {
                const __m128i qs8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(in_blk.qs + i));
                const __m128i gamma16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(gamma_block + i));
                const __m256i qs32 = _mm256_cvtepi8_epi32(qs8);
                const __m256i gamma32 = _mm256_cvtepi16_epi32(gamma16);
                const __m256i prod32 = _mm256_mullo_epi32(qs32, gamma32);
                _mm256_store_si256(reinterpret_cast<__m256i *>(prod + i), prod32);
                max_abs_v = _mm256_max_epi32(max_abs_v, _mm256_abs_epi32(prod32));
            }

            const int32_t max_abs = reduce_max_i32_avx2(max_abs_v);
            const float max_prod = static_cast<float>(max_abs);
            const float d_in = fp16_to_fp32_q8(in_blk.d);
            const float d_out = (max_abs > 0)
                                    ? (d_in * max_prod * inv_rms_float * (1.0f / (127.0f * 2048.0f)))
                                    : 0.0f;
            const float requant_scale = (max_abs > 0) ? (127.0f / max_prod) : 0.0f;
            const __m256 scale_v = _mm256_set1_ps(requant_scale);
            const __m256i lo_v = _mm256_set1_epi32(-127);
            const __m256i hi_v = _mm256_set1_epi32(127);

            int32_t sum_qs = 0;
            for (int i = 0; i < 32; i += 8)
            {
                const __m256i prod32 = _mm256_load_si256(reinterpret_cast<const __m256i *>(prod + i));
                __m256i q32 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(prod32), scale_v));
                q32 = _mm256_max_epi32(lo_v, _mm256_min_epi32(hi_v, q32));
                sum_qs += reduce_add_i32_avx2(q32);

                const __m128i q16 = _mm_packs_epi32(
                    _mm256_castsi256_si128(q32),
                    _mm256_extracti128_si256(q32, 1));
                const __m128i q8 = _mm_packs_epi16(q16, _mm_setzero_si128());
                _mm_storel_epi64(reinterpret_cast<__m128i *>(out_blk.qs + i), q8);
            }

            out_blk.sum_qs = static_cast<int16_t>(sum_qs);
            out_blk.d = fp32_to_fp16_q8(d_out);
        }
#endif

        __attribute__((always_inline)) inline void rmsnorm_q8_1_apply_block_float_scaled_scalar(
            const Q8_1Block &in_blk,
            const int16_t *gamma_block,
            Q8_1Block &out_blk,
            float inv_rms_float)
        {
            int32_t prod[32];
            int32_t max_abs = 0;
            for (int i = 0; i < 32; ++i)
            {
                prod[i] = static_cast<int32_t>(in_blk.qs[i]) * static_cast<int32_t>(gamma_block[i]);
                max_abs = std::max(max_abs, std::abs(prod[i]));
            }

            const float max_prod = static_cast<float>(max_abs);
            const float d_in = fp16_to_fp32_q8(in_blk.d);
            const float d_out = (max_abs > 0)
                                    ? (d_in * max_prod * inv_rms_float * (1.0f / (127.0f * 2048.0f)))
                                    : 0.0f;
            const float requant_scale = (max_abs > 0) ? (127.0f / max_prod) : 0.0f;

            int32_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::lrint(static_cast<float>(prod[i]) * requant_scale));
                q = (q > 127) ? 127 : (q < -127) ? -127
                                                 : q;
                out_blk.qs[i] = static_cast<int8_t>(q);
                sum_qs += q;
            }

            out_blk.sum_qs = static_cast<int16_t>(sum_qs);
            out_blk.d = fp32_to_fp16_q8(d_out);
        }

        __attribute__((always_inline)) inline void rmsnorm_q8_1_apply_block_float_scaled(
            const Q8_1Block &in_blk,
            const int16_t *gamma_block,
            Q8_1Block &out_blk,
            float inv_rms_float)
        {
#if defined(__AVX2__)
            rmsnorm_q8_1_apply_block_float_scaled_avx2(in_blk, gamma_block, out_blk, inv_rms_float);
#else
            rmsnorm_q8_1_apply_block_float_scaled_scalar(in_blk, gamma_block, out_blk, inv_rms_float);
#endif
        }
    }

    /**
     * @brief Unrolled (2x) Single-Pass RMSNorm for Q8_1
     *
     * Optimizes the single-pass approach by processing 2 blocks per iteration.
     * This hides the latency of the max_abs -> scale dependency chain by
     * interleaving instructions for two independent blocks.
     *
     * Maintains strict per-block correctness (unlike the global-scale two-pass approach).
     */
    void rmsnorm_q8_1_pure_integer_row_unrolled(
        const Q8_1Block *input,
        const int16_t *gamma_q12,
        Q8_1Block *output,
        std::size_t blocks_per_row,
        int32_t epsilon_scaled)
    {
        if (!input || !gamma_q12 || !output || blocks_per_row == 0)
            return;
        (void)epsilon_scaled;

        const size_t cols = blocks_per_row * 32;

        // ====================================================================
        // Phase 1 & 2: Compute Sum(d² * sum_qs²) using vector accumulation
        // ====================================================================

#if defined(__AVX512F__)
        __m512 v_total_sum = _mm512_setzero_ps();

        // Unrolled sum loop (4x)
        size_t b = 0;
        for (; b + 4 <= blocks_per_row; b += 4)
        {
            // Block 0
            float d0 = fp16_to_fp32_q8(input[b].d);
            __m512 v_d0_sq = _mm512_set1_ps(d0 * d0);
            __m256i qs8_0 = _mm256_loadu_si256((const __m256i *)input[b].qs);
            __m512i qs16_0 = _mm512_cvtepi8_epi16(qs8_0);
            __m512i qs_sq0 = _mm512_madd_epi16(qs16_0, qs16_0);
            v_total_sum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(qs_sq0), v_d0_sq, v_total_sum);

            // Block 1
            float d1 = fp16_to_fp32_q8(input[b + 1].d);
            __m512 v_d1_sq = _mm512_set1_ps(d1 * d1);
            __m256i qs8_1 = _mm256_loadu_si256((const __m256i *)input[b + 1].qs);
            __m512i qs16_1 = _mm512_cvtepi8_epi16(qs8_1);
            __m512i qs_sq1 = _mm512_madd_epi16(qs16_1, qs16_1);
            v_total_sum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(qs_sq1), v_d1_sq, v_total_sum);

            // Block 2
            float d2 = fp16_to_fp32_q8(input[b + 2].d);
            __m512 v_d2_sq = _mm512_set1_ps(d2 * d2);
            __m256i qs8_2 = _mm256_loadu_si256((const __m256i *)input[b + 2].qs);
            __m512i qs16_2 = _mm512_cvtepi8_epi16(qs8_2);
            __m512i qs_sq2 = _mm512_madd_epi16(qs16_2, qs16_2);
            v_total_sum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(qs_sq2), v_d2_sq, v_total_sum);

            // Block 3
            float d3 = fp16_to_fp32_q8(input[b + 3].d);
            __m512 v_d3_sq = _mm512_set1_ps(d3 * d3);
            __m256i qs8_3 = _mm256_loadu_si256((const __m256i *)input[b + 3].qs);
            __m512i qs16_3 = _mm512_cvtepi8_epi16(qs8_3);
            __m512i qs_sq3 = _mm512_madd_epi16(qs16_3, qs16_3);
            v_total_sum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(qs_sq3), v_d3_sq, v_total_sum);
        }

        double total_sum_sq = static_cast<double>(_mm512_reduce_add_ps(v_total_sum));
#else
        double total_sum_sq = 0.0;
        size_t b = 0;
#endif
        for (; b < blocks_per_row; ++b)
        {
            float d = fp16_to_fp32_q8(input[b].d);
            int32_t s = 0;
#if defined(__AVX512F__)
            __m256i qs_8 = _mm256_loadu_si256((const __m256i *)input[b].qs);
            __m512i qs_16 = _mm512_cvtepi8_epi16(qs_8);
            __m512i qs_sq = _mm512_madd_epi16(qs_16, qs_16);
            s = _mm512_reduce_add_epi32(qs_sq);
#else
            for (int i = 0; i < 32; ++i)
                s += input[b].qs[i] * input[b].qs[i];
#endif
            total_sum_sq += static_cast<double>(d) * d * s;
        }

        // ====================================================================
        // Phase 3: Compute RMS and inverse RMS
        // ====================================================================
        double mean_sq = total_sum_sq / cols;
        mean_sq += 1e-5; // Epsilon
        double rms = std::sqrt(mean_sq);
        const float inv_rms_float = 1.0f / static_cast<float>(rms);

        // ====================================================================
        // Phase 4: Apply normalization (Unrolled 2x)
        // ====================================================================

#if defined(__AVX512F__)
        // Optimized implementation using float scaling to avoid 64-bit integer ops
        __m512 v_inv_rms_f = _mm512_set1_ps(inv_rms_float);
        __m512i v_127_i = _mm512_set1_epi32(127);
        __m512i v_neg127_i = _mm512_set1_epi32(-127);
        __m512i v_abs_mask = _mm512_set1_epi32(0x7FFFFFFF);

        // Unrolled loop (4x) with Vectorized Division
        b = 0;
        for (; b + 4 <= blocks_per_row; b += 4)
        {
            // Block 0
            const Q8_1Block &in0 = input[b];
            __m256i qs8_0 = _mm256_loadu_si256((const __m256i *)in0.qs);
            __m512i qs16_0 = _mm512_cvtepi8_epi16(qs8_0);
            __m512i g16_0 = _mm512_loadu_si512((const __m512i *)(gamma_q12 + b * 32));

            __m512i qs32_lo_0 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(qs16_0));
            __m512i qs32_hi_0 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(qs16_0, 1));
            __m512i g32_lo_0 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(g16_0));
            __m512i g32_hi_0 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(g16_0, 1));

            __m512i prod32_lo_0 = _mm512_mullo_epi32(qs32_lo_0, g32_lo_0);
            __m512i prod32_hi_0 = _mm512_mullo_epi32(qs32_hi_0, g32_hi_0);

            __m512i max_int_v_0 = _mm512_max_epi32(_mm512_abs_epi32(prod32_lo_0), _mm512_abs_epi32(prod32_hi_0));
            float m0 = (float)_mm512_reduce_max_epi32(max_int_v_0);

            // Block 1
            const Q8_1Block &in1 = input[b + 1];
            __m256i qs8_1 = _mm256_loadu_si256((const __m256i *)in1.qs);
            __m512i qs16_1 = _mm512_cvtepi8_epi16(qs8_1);
            __m512i g16_1 = _mm512_loadu_si512((const __m512i *)(gamma_q12 + (b + 1) * 32));

            __m512i qs32_lo_1 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(qs16_1));
            __m512i qs32_hi_1 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(qs16_1, 1));
            __m512i g32_lo_1 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(g16_1));
            __m512i g32_hi_1 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(g16_1, 1));

            __m512i prod32_lo_1 = _mm512_mullo_epi32(qs32_lo_1, g32_lo_1);
            __m512i prod32_hi_1 = _mm512_mullo_epi32(qs32_hi_1, g32_hi_1);

            __m512i max_int_v_1 = _mm512_max_epi32(_mm512_abs_epi32(prod32_lo_1), _mm512_abs_epi32(prod32_hi_1));
            float m1 = (float)_mm512_reduce_max_epi32(max_int_v_1);

            // Block 2
            const Q8_1Block &in2 = input[b + 2];
            __m256i qs8_2 = _mm256_loadu_si256((const __m256i *)in2.qs);
            __m512i qs16_2 = _mm512_cvtepi8_epi16(qs8_2);
            __m512i g16_2 = _mm512_loadu_si512((const __m512i *)(gamma_q12 + (b + 2) * 32));

            __m512i qs32_lo_2 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(qs16_2));
            __m512i qs32_hi_2 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(qs16_2, 1));
            __m512i g32_lo_2 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(g16_2));
            __m512i g32_hi_2 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(g16_2, 1));

            __m512i prod32_lo_2 = _mm512_mullo_epi32(qs32_lo_2, g32_lo_2);
            __m512i prod32_hi_2 = _mm512_mullo_epi32(qs32_hi_2, g32_hi_2);

            __m512i max_int_v_2 = _mm512_max_epi32(_mm512_abs_epi32(prod32_lo_2), _mm512_abs_epi32(prod32_hi_2));
            float m2 = (float)_mm512_reduce_max_epi32(max_int_v_2);

            // Block 3
            const Q8_1Block &in3 = input[b + 3];
            __m256i qs8_3 = _mm256_loadu_si256((const __m256i *)in3.qs);
            __m512i qs16_3 = _mm512_cvtepi8_epi16(qs8_3);
            __m512i g16_3 = _mm512_loadu_si512((const __m512i *)(gamma_q12 + (b + 3) * 32));

            __m512i qs32_lo_3 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(qs16_3));
            __m512i qs32_hi_3 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(qs16_3, 1));
            __m512i g32_lo_3 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(g16_3));
            __m512i g32_hi_3 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(g16_3, 1));

            __m512i prod32_lo_3 = _mm512_mullo_epi32(qs32_lo_3, g32_lo_3);
            __m512i prod32_hi_3 = _mm512_mullo_epi32(qs32_hi_3, g32_hi_3);

            __m512i max_int_v_3 = _mm512_max_epi32(_mm512_abs_epi32(prod32_lo_3), _mm512_abs_epi32(prod32_hi_3));
            float m3 = (float)_mm512_reduce_max_epi32(max_int_v_3);

            // Vectorized Reciprocal
            __m128 v_max = _mm_set_ps(m3, m2, m1, m0);
            v_max = _mm_max_ps(v_max, _mm_set1_ps(1e-9f)); // Avoid div/0
            __m128 v_rcp = _mm_rcp_ps(v_max);
            // Newton-Raphson: x1 = x0 * (2 - a * x0)
            v_rcp = _mm_mul_ps(v_rcp, _mm_sub_ps(_mm_set1_ps(2.0f), _mm_mul_ps(v_max, v_rcp)));
            __m128 v_scale = _mm_mul_ps(v_rcp, _mm_set1_ps(127.0f));

            // Extract scales
            float s0 = _mm_cvtss_f32(v_scale);
            float s1 = _mm_cvtss_f32(_mm_shuffle_ps(v_scale, v_scale, 0x55));
            float s2 = _mm_cvtss_f32(_mm_shuffle_ps(v_scale, v_scale, 0xAA));
            float s3 = _mm_cvtss_f32(_mm_shuffle_ps(v_scale, v_scale, 0xFF));

            // Finish Block 0
            {
                Q8_1Block &out = output[b];
                float d_in = fp16_to_fp32_q8(in0.d);
                float d_out_val = d_in * m0 * inv_rms_float * (1.0f / (127.0f * 2048.0f));
                out.d = fp32_to_fp16_q8(d_out_val);

                __m512 v_s0 = _mm512_set1_ps(s0);
                __m512 prod_ps_lo = _mm512_cvtepi32_ps(prod32_lo_0);
                __m512 prod_ps_hi = _mm512_cvtepi32_ps(prod32_hi_0);
                __m512i final_i_lo = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_lo, v_s0));
                __m512i final_i_hi = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_hi, v_s0));

                final_i_lo = _mm512_max_epi32(final_i_lo, v_neg127_i);
                final_i_lo = _mm512_min_epi32(final_i_lo, v_127_i);
                final_i_hi = _mm512_max_epi32(final_i_hi, v_neg127_i);
                final_i_hi = _mm512_min_epi32(final_i_hi, v_127_i);

                __m128i out8_lo = _mm512_cvtepi32_epi8(final_i_lo);
                __m128i out8_hi = _mm512_cvtepi32_epi8(final_i_hi);
                __m256i out8 = _mm256_inserti128_si256(_mm256_castsi128_si256(out8_lo), out8_hi, 1);
                _mm256_storeu_si256((__m256i *)out.qs, out8);
                out.sum_qs = static_cast<int16_t>(_mm512_reduce_add_epi32(final_i_lo) + _mm512_reduce_add_epi32(final_i_hi));
            }

            // Finish Block 1
            {
                Q8_1Block &out = output[b + 1];
                float d_in = fp16_to_fp32_q8(in1.d);
                float d_out_val = d_in * m1 * inv_rms_float * (1.0f / (127.0f * 2048.0f));
                out.d = fp32_to_fp16_q8(d_out_val);

                __m512 v_s1 = _mm512_set1_ps(s1);
                __m512 prod_ps_lo = _mm512_cvtepi32_ps(prod32_lo_1);
                __m512 prod_ps_hi = _mm512_cvtepi32_ps(prod32_hi_1);
                __m512i final_i_lo = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_lo, v_s1));
                __m512i final_i_hi = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_hi, v_s1));

                final_i_lo = _mm512_max_epi32(final_i_lo, v_neg127_i);
                final_i_lo = _mm512_min_epi32(final_i_lo, v_127_i);
                final_i_hi = _mm512_max_epi32(final_i_hi, v_neg127_i);
                final_i_hi = _mm512_min_epi32(final_i_hi, v_127_i);

                __m128i out8_lo = _mm512_cvtepi32_epi8(final_i_lo);
                __m128i out8_hi = _mm512_cvtepi32_epi8(final_i_hi);
                __m256i out8 = _mm256_inserti128_si256(_mm256_castsi128_si256(out8_lo), out8_hi, 1);
                _mm256_storeu_si256((__m256i *)out.qs, out8);
                out.sum_qs = static_cast<int16_t>(_mm512_reduce_add_epi32(final_i_lo) + _mm512_reduce_add_epi32(final_i_hi));
            }

            // Finish Block 2
            {
                Q8_1Block &out = output[b + 2];
                float d_in = fp16_to_fp32_q8(in2.d);
                float d_out_val = d_in * m2 * inv_rms_float * (1.0f / (127.0f * 2048.0f));
                out.d = fp32_to_fp16_q8(d_out_val);

                __m512 v_s2 = _mm512_set1_ps(s2);
                __m512 prod_ps_lo = _mm512_cvtepi32_ps(prod32_lo_2);
                __m512 prod_ps_hi = _mm512_cvtepi32_ps(prod32_hi_2);
                __m512i final_i_lo = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_lo, v_s2));
                __m512i final_i_hi = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_hi, v_s2));

                final_i_lo = _mm512_max_epi32(final_i_lo, v_neg127_i);
                final_i_lo = _mm512_min_epi32(final_i_lo, v_127_i);
                final_i_hi = _mm512_max_epi32(final_i_hi, v_neg127_i);
                final_i_hi = _mm512_min_epi32(final_i_hi, v_127_i);

                __m128i out8_lo = _mm512_cvtepi32_epi8(final_i_lo);
                __m128i out8_hi = _mm512_cvtepi32_epi8(final_i_hi);
                __m256i out8 = _mm256_inserti128_si256(_mm256_castsi128_si256(out8_lo), out8_hi, 1);
                _mm256_storeu_si256((__m256i *)out.qs, out8);
                out.sum_qs = static_cast<int16_t>(_mm512_reduce_add_epi32(final_i_lo) + _mm512_reduce_add_epi32(final_i_hi));
            }

            // Finish Block 3
            {
                Q8_1Block &out = output[b + 3];
                float d_in = fp16_to_fp32_q8(in3.d);
                float d_out_val = d_in * m3 * inv_rms_float * (1.0f / (127.0f * 2048.0f));
                out.d = fp32_to_fp16_q8(d_out_val);

                __m512 v_s3 = _mm512_set1_ps(s3);
                __m512 prod_ps_lo = _mm512_cvtepi32_ps(prod32_lo_3);
                __m512 prod_ps_hi = _mm512_cvtepi32_ps(prod32_hi_3);
                __m512i final_i_lo = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_lo, v_s3));
                __m512i final_i_hi = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_hi, v_s3));

                final_i_lo = _mm512_max_epi32(final_i_lo, v_neg127_i);
                final_i_lo = _mm512_min_epi32(final_i_lo, v_127_i);
                final_i_hi = _mm512_max_epi32(final_i_hi, v_neg127_i);
                final_i_hi = _mm512_min_epi32(final_i_hi, v_127_i);

                __m128i out8_lo = _mm512_cvtepi32_epi8(final_i_lo);
                __m128i out8_hi = _mm512_cvtepi32_epi8(final_i_hi);
                __m256i out8 = _mm256_inserti128_si256(_mm256_castsi128_si256(out8_lo), out8_hi, 1);
                _mm256_storeu_si256((__m256i *)out.qs, out8);
                out.sum_qs = static_cast<int16_t>(_mm512_reduce_add_epi32(final_i_lo) + _mm512_reduce_add_epi32(final_i_hi));
            }
        }
        // Remaining blocks
        for (; b < blocks_per_row; ++b)
        {
            const Q8_1Block &in = input[b];
            Q8_1Block &out = output[b];
            float d_in = fp16_to_fp32_q8(in.d);

            __m256i qs8 = _mm256_loadu_si256((const __m256i *)in.qs);
            __m512i qs16 = _mm512_cvtepi8_epi16(qs8);
            __m512i g16 = _mm512_loadu_si512((const __m512i *)(gamma_q12 + b * 32));

            __m512i qs32_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(qs16));
            __m512i qs32_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(qs16, 1));
            __m512i g32_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(g16));
            __m512i g32_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(g16, 1));

            __m512i prod32_lo = _mm512_mullo_epi32(qs32_lo, g32_lo);
            __m512i prod32_hi = _mm512_mullo_epi32(qs32_hi, g32_hi);

            __m512i abs_int_lo = _mm512_abs_epi32(prod32_lo);
            __m512i abs_int_hi = _mm512_abs_epi32(prod32_hi);
            __m512i max_int_v = _mm512_max_epi32(abs_int_lo, abs_int_hi);
            int32_t max_int = _mm512_reduce_max_epi32(max_int_v);
            float max_prod = static_cast<float>(max_int);

            float d_out_val = (max_prod > 1e-9f) ? (d_in * max_prod * inv_rms_float / (127.0f * 2048.0f)) : 0.0f;
            out.d = fp32_to_fp16_q8(d_out_val);

            float combined_scale_val = (max_prod > 1e-9f) ? (127.0f / max_prod) : 0.0f;
            __m512 v_combined_scale = _mm512_set1_ps(combined_scale_val);

            __m512 prod_ps_lo = _mm512_cvtepi32_ps(prod32_lo);
            __m512 prod_ps_hi = _mm512_cvtepi32_ps(prod32_hi);
            __m512i final_i_lo = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_lo, v_combined_scale));
            __m512i final_i_hi = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_hi, v_combined_scale));

            final_i_lo = _mm512_max_epi32(final_i_lo, v_neg127_i);
            final_i_lo = _mm512_min_epi32(final_i_lo, v_127_i);
            final_i_hi = _mm512_max_epi32(final_i_hi, v_neg127_i);
            final_i_hi = _mm512_min_epi32(final_i_hi, v_127_i);

            __m128i out8_lo = _mm512_cvtepi32_epi8(final_i_lo);
            __m128i out8_hi = _mm512_cvtepi32_epi8(final_i_hi);
            __m256i out8 = _mm256_inserti128_si256(_mm256_castsi128_si256(out8_lo), out8_hi, 1);
            _mm256_storeu_si256((__m256i *)out.qs, out8);
            out.sum_qs = static_cast<int16_t>(_mm512_reduce_add_epi32(final_i_lo) + _mm512_reduce_add_epi32(final_i_hi));
        }
#endif
        for (; b < blocks_per_row; ++b)
        {
            rmsnorm_q8_1_apply_block_float_scaled(input[b], gamma_q12 + b * 32, output[b], inv_rms_float);
        }
    }

    void rmsnorm_q8_1_pure_integer_row(
        const Q8_1Block *input,
        const int16_t *gamma_q12, // Pre-quantized gamma in Q11 format (clamp(gamma, -16, 16) * 2047)
        Q8_1Block *output,
        std::size_t blocks_per_row,
        int32_t epsilon_scaled) // Epsilon pre-scaled appropriately
    {
        if (!input || !gamma_q12 || !output || blocks_per_row == 0)
            return;
        (void)epsilon_scaled;

        const size_t cols = blocks_per_row * 32;

        // ====================================================================
        // Phase 1 & 2: Compute Sum(d² * sum_qs²) using double precision
        // This avoids overflow/underflow issues with fixed-point accumulation
        // ====================================================================
        double total_sum_sq = 0.0;

        for (size_t b = 0; b < blocks_per_row; ++b)
        {
            float d = fp16_to_fp32_q8(input[b].d);
            int32_t s = 0;

#if defined(__AVX512F__)
            __m256i qs_8 = _mm256_loadu_si256((const __m256i *)input[b].qs);
            __m512i qs_16 = _mm512_cvtepi8_epi16(qs_8);
            __m512i qs_sq = _mm512_madd_epi16(qs_16, qs_16);
            s = _mm512_reduce_add_epi32(qs_sq);
#else
            for (int i = 0; i < 32; ++i)
                s += input[b].qs[i] * input[b].qs[i];
#endif
            total_sum_sq += static_cast<double>(d) * d * s;
        }

        // ====================================================================
        // Phase 3: Compute RMS and inverse RMS
        // ====================================================================

        // Mean = Total / Cols
        double mean_sq = total_sum_sq / cols;
        mean_sq += 1e-5; // Epsilon

        // Compute 1/sqrt(mean_qs_sq) in Q24 format
        // inv_rms = 2^24 / sqrt(mean_sq)
        double rms = std::sqrt(mean_sq);
#if defined(__AVX512F__)
        double inv_rms_f = 16777216.0 / rms;

        // Clamp to uint32 range to avoid overflow in integer loop
        if (inv_rms_f > 2147483647.0)
            inv_rms_f = 2147483647.0;

        uint32_t inv_rms_q24 = static_cast<uint32_t>(inv_rms_f);
        if (inv_rms_q24 == 0)
            inv_rms_q24 = 1;
#else
        const float inv_rms_float = 1.0f / static_cast<float>(rms);
#endif

        // ====================================================================
        // Phase 4: Apply normalization (vectorized)
        // For each block:
        //   1) Compute y_high = qs * gamma * inv_rms (64-bit precision)
        //      y_high scale: Q0 * Q12 * (2^14/RMS) = Q26 / RMS
        //   2) y = y_high >> 20 (Q6 / RMS)
        //   3) Find max_abs = max(|y|) across the block
        //   4) d_out = d_in * max_abs / (127 * 64)
        //   5) qs_out = clamp(y * 127 / max_abs, -127, 127)
        // ====================================================================

#if defined(__AVX512F__)
        __m512i v_inv_rms = _mm512_set1_epi32(inv_rms_q24);
        // Permutation indices for interleaving even/odd results
        __m512i v_perm_idx = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);

        for (size_t b = 0; b < blocks_per_row; ++b)
        {
            const Q8_1Block &in_blk = input[b];
            Q8_1Block &out_blk = output[b];
            float d_in = fp16_to_fp32_q8(in_blk.d);

            // Load and expand qs to int32
            __m128i qs_lo = _mm_loadu_si128((const __m128i *)in_blk.qs);
            __m128i qs_hi = _mm_loadu_si128((const __m128i *)(in_blk.qs + 16));
            __m512i qs32_lo = _mm512_cvtepi8_epi32(qs_lo);
            __m512i qs32_hi = _mm512_cvtepi8_epi32(qs_hi);

            // Load gamma (int16) -> int32
            __m256i gamma16_lo = _mm256_loadu_si256((const __m256i *)(gamma_q12 + b * 32));
            __m256i gamma16_hi = _mm256_loadu_si256((const __m256i *)(gamma_q12 + b * 32 + 16));
            __m512i gamma32_lo = _mm512_cvtepi16_epi32(gamma16_lo);
            __m512i gamma32_hi = _mm512_cvtepi16_epi32(gamma16_hi);

            // Compute term = qs * gamma (int32)
            __m512i term_lo = _mm512_mullo_epi32(qs32_lo, gamma32_lo);
            __m512i term_hi = _mm512_mullo_epi32(qs32_hi, gamma32_hi);

            // Compute res = term * inv_rms (64-bit)
            // Even lanes
            __m512i res_lo_even = _mm512_mul_epi32(term_lo, v_inv_rms);
            __m512i res_hi_even = _mm512_mul_epi32(term_hi, v_inv_rms);
            // Odd lanes (shift term right by 32)
            __m512i res_lo_odd = _mm512_mul_epi32(_mm512_srli_epi64(term_lo, 32), v_inv_rms);
            __m512i res_hi_odd = _mm512_mul_epi32(_mm512_srli_epi64(term_hi, 32), v_inv_rms);

            // Shift right by 20 to get Q6 result (fits in int32)
            __m512i y_lo_even = _mm512_srai_epi64(res_lo_even, 20);
            __m512i y_hi_even = _mm512_srai_epi64(res_hi_even, 20);
            __m512i y_lo_odd = _mm512_srai_epi64(res_lo_odd, 20);
            __m512i y_hi_odd = _mm512_srai_epi64(res_hi_odd, 20);

            // Pack back to int32
            __m256i y256_lo_even = _mm512_cvtepi64_epi32(y_lo_even);
            __m256i y256_lo_odd = _mm512_cvtepi64_epi32(y_lo_odd);
            __m256i y256_hi_even = _mm512_cvtepi64_epi32(y_hi_even);
            __m256i y256_hi_odd = _mm512_cvtepi64_epi32(y_hi_odd);

            // Interleave even/odd results
            __m512i y_lo = _mm512_castsi256_si512(y256_lo_even);
            y_lo = _mm512_inserti64x4(y_lo, y256_lo_odd, 1);
            y_lo = _mm512_permutexvar_epi32(v_perm_idx, y_lo);

            __m512i y_hi = _mm512_castsi256_si512(y256_hi_even);
            y_hi = _mm512_inserti64x4(y_hi, y256_hi_odd, 1);
            y_hi = _mm512_permutexvar_epi32(v_perm_idx, y_hi);

            // Find max absolute value
            __m512i abs_lo = _mm512_abs_epi32(y_lo);
            __m512i abs_hi = _mm512_abs_epi32(y_hi);
            __m512i abs_max_vec = _mm512_max_epi32(abs_lo, abs_hi);
            int32_t max_abs = _mm512_reduce_max_epi32(abs_max_vec);

            // Compute output scale: d_out = d_in * max_abs / (127 * 65536)
            // 65536 comes from the Q16 nature of y (result of >> 20 from Q36)
            float d_out;
            if (max_abs <= 127)
            {
                if (max_abs == 0)
                {
                    d_out = 0.0f;
                }
                else
                {
                    d_out = d_in * static_cast<float>(max_abs) / (127.0f * 32768.0f);
                }
            }
            else
            {
                d_out = d_in * static_cast<float>(max_abs) / (127.0f * 32768.0f);
            }

            // Rescale y values: y_rescaled = y * 127 / max_abs
            if (max_abs > 0)
            {
                __m512 scale_factor = _mm512_set1_ps(127.0f / static_cast<float>(max_abs));
                __m512 y_f32_lo = _mm512_cvtepi32_ps(y_lo);
                __m512 y_f32_hi = _mm512_cvtepi32_ps(y_hi);
                y_f32_lo = _mm512_mul_ps(y_f32_lo, scale_factor);
                y_f32_hi = _mm512_mul_ps(y_f32_hi, scale_factor);
                y_lo = _mm512_cvtps_epi32(y_f32_lo);
                y_hi = _mm512_cvtps_epi32(y_f32_hi);
            }

            // Clamp to [-127, 127]
            __m512i v_127 = _mm512_set1_epi32(127);
            __m512i v_neg127 = _mm512_set1_epi32(-127);
            y_lo = _mm512_max_epi32(y_lo, v_neg127);
            y_lo = _mm512_min_epi32(y_lo, v_127);
            y_hi = _mm512_max_epi32(y_hi, v_neg127);
            y_hi = _mm512_min_epi32(y_hi, v_127);

            // Pack back to int8
            __m128i out8_lo = _mm512_cvtepi32_epi8(y_lo);
            __m128i out8_hi = _mm512_cvtepi32_epi8(y_hi);

            _mm_storeu_si128((__m128i *)out_blk.qs, out8_lo);
            _mm_storeu_si128((__m128i *)(out_blk.qs + 16), out8_hi);

            // Compute sum_qs
            int32_t sum_lo = _mm512_reduce_add_epi32(y_lo);
            int32_t sum_hi = _mm512_reduce_add_epi32(y_hi);
            out_blk.sum_qs = static_cast<int16_t>(sum_lo + sum_hi);

            // Store output scale
            out_blk.d = fp32_to_fp16_q8(d_out);
        }
#else
        for (size_t b = 0; b < blocks_per_row; ++b)
        {
            rmsnorm_q8_1_apply_block_float_scaled(input[b], gamma_q12 + b * 32, output[b], inv_rms_float);
        }
#endif
    }

    void quantize_rmsnorm_gamma_q11(const float *gamma, int16_t *gamma_q11, std::size_t cols)
    {
        if (cols == 0)
            return;
        if (!gamma || !gamma_q11)
            throw std::invalid_argument("Q11 gamma quantization requires complete gamma and output rows");
        const uintptr_t gamma_begin = reinterpret_cast<uintptr_t>(gamma);
        const uintptr_t output_begin = reinterpret_cast<uintptr_t>(gamma_q11);
        if (gamma_begin + cols * sizeof(float) > output_begin &&
            output_begin + cols * sizeof(int16_t) > gamma_begin)
            throw std::invalid_argument("Q11 gamma quantization requires disjoint gamma and output rows");
        // Q11 with a [-16, 16] range keeps Qwen2.5 gamma values above 13.0
        // representable; the former Q12 [-8, 8] range clipped them.
#ifdef __AVX512F__
        {
            size_t i = 0;
            const __m512 scale = _mm512_set1_ps(2047.0f); // Q11 scale for [-16, 16] range
            const __m512 clamp_lo = _mm512_set1_ps(-16.0f);
            const __m512 clamp_hi = _mm512_set1_ps(16.0f);
            const __m512 int16_lo = _mm512_set1_ps(-32767.0f);
            const __m512 int16_hi = _mm512_set1_ps(32767.0f);

            for (; i + 32 <= cols; i += 32)
            {
                __m512 v0 = _mm512_loadu_ps(gamma + i);
                __m512 v1 = _mm512_loadu_ps(gamma + i + 16);
                v0 = _mm512_max_ps(v0, clamp_lo);
                v0 = _mm512_min_ps(v0, clamp_hi);
                v1 = _mm512_max_ps(v1, clamp_lo);
                v1 = _mm512_min_ps(v1, clamp_hi);
                v0 = _mm512_mul_ps(v0, scale);
                v1 = _mm512_mul_ps(v1, scale);
                v0 = _mm512_max_ps(v0, int16_lo);
                v0 = _mm512_min_ps(v0, int16_hi);
                v1 = _mm512_max_ps(v1, int16_lo);
                v1 = _mm512_min_ps(v1, int16_hi);
                __m512i i0 = _mm512_cvtps_epi32(v0);
                __m512i i1 = _mm512_cvtps_epi32(v1);
                __m512i packed = _mm512_packs_epi32(i0, i1);
                packed = _mm512_permutexvar_epi64(
                    _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), packed);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(gamma_q11 + i), packed);
            }
            for (; i < cols; ++i)
            {
                float clamped = std::max(-16.0f, std::min(16.0f, gamma[i]));
                float scaled = clamped * 2047.0f;
                scaled = std::max(-32767.0f, std::min(32767.0f, scaled));
                gamma_q11[i] = static_cast<int16_t>(scaled);
            }
        }
#else
        for (size_t i = 0; i < cols; ++i)
        {
            float clamped = std::max(-16.0f, std::min(16.0f, gamma[i]));
            float scaled = clamped * 2047.0f;
            scaled = std::max(-32767.0f, std::min(32767.0f, scaled));
            gamma_q11[i] = static_cast<int16_t>(scaled);
        }
#endif
    }

    const int16_t *rmsnorm_gamma_q11_for_owner(
        const float *gamma, std::size_t cols, const std::weak_ptr<const void> &owner)
    {
        struct Entry
        {
            std::weak_ptr<const void> owner;
            const float *gamma = nullptr;
            std::size_t cols = 0;
            std::vector<int16_t> words;
        };
        // Single writer: every thread owns its entry. The weak reference pins
        // the owner's control block, so owner equivalence cannot be satisfied
        // by a different, later allocation that reuses the gamma address.
        thread_local Entry entry;
        const std::weak_ptr<const void> unowned;
        const bool has_owner = owner.owner_before(unowned) || unowned.owner_before(owner);
        const bool same_owner = !entry.owner.owner_before(owner) && !owner.owner_before(entry.owner);
        if (has_owner && !owner.expired() && same_owner && entry.gamma == gamma && entry.cols == cols)
            return entry.words.data();

        if (entry.words.size() < cols)
            entry.words.resize(cols);
        quantize_rmsnorm_gamma_q11(gamma, entry.words.data(), cols);
        // Unowned or expired weights have no authentic identity and stay uncached.
        const bool cacheable = has_owner && !owner.expired();
        entry.owner = cacheable ? owner : std::weak_ptr<const void>{};
        entry.gamma = cacheable ? gamma : nullptr;
        entry.cols = cacheable ? cols : 0;
        return entry.words.data();
    }

    void rmsnorm_q8_1_pure_integer(
        const Q8_1Block *input,
        const int16_t *gamma_q11,
        Q8_1Block *output,
        std::size_t rows,
        std::size_t blocks_per_row,
        float epsilon,
        const RMSNormExecOptions &opts)
    {
        if (!input || !gamma_q11 || !output || rows == 0 || blocks_per_row == 0)
            return;

        // Scale epsilon to integer domain
        // Typical epsilon is 1e-5 to 1e-6. We scale to ~1-100 range.
        int32_t epsilon_scaled = static_cast<int32_t>(epsilon * 1e10f);
        if (epsilon_scaled < 1)
            epsilon_scaled = 1;

        // Process rows - parallelize if we have enough work
        // Each row does: sumsq -> inv_rms -> apply (all in rmsnorm_q8_1_pure_integer_row)
        // ================================================================
        // SMOOTH DYNAMIC THREADING (Dec 2025)
        //
        // Single heuristic: threads = total_bytes / bytes_per_thread
        // This naturally scales from 1 thread (small) to many (large).
        // No hard cutoffs - the math handles it smoothly.
        // ================================================================
        bool should_parallelize = opts.allow_parallel;

        if (should_parallelize)
        {
            // Use OMP_WORKSHARE_REGION for nested-safe execution
            auto do_work = [&]()
            {
#pragma omp for schedule(static)
                for (long long r = 0; r < static_cast<long long>(rows); ++r)
                {
                    rmsnorm_q8_1_pure_integer_row_unrolled(
                        input + r * blocks_per_row,
                        gamma_q11,
                        output + r * blocks_per_row,
                        blocks_per_row,
                        epsilon_scaled);
                }
            };
            OMP_WORKSHARE_REGION(do_work);
            return;
        }

        // Sequential path
        for (size_t r = 0; r < rows; ++r)
        {
            rmsnorm_q8_1_pure_integer_row_unrolled(
                input + r * blocks_per_row,
                gamma_q11,
                output + r * blocks_per_row,
                blocks_per_row,
                epsilon_scaled);
        }
    }

    // ========================================================================
    // Q16_1 RMSNorm (Q16_1 input → FP32 output)
    // ========================================================================

#if defined(__AVX2__)
    /** @brief Native AVX2 Q16 program with FP32 squares and eight fixed FP64 chains.
     * @param input Complete original Q16 row; @param gamma Immutable original weights.
     * @param output Disjoint floating row; @param cols Complete32-value geometry.
     * @param epsilon Original finite nonnegative policy.
     * @throws std::invalid_argument For malformed original spans/aliases/policy.
     * @details Nonfinite summary or undefined inverse is fatal through the
     * release-safe sink, never an exception across an OpenMP boundary.
     */
    static void rmsnorm_q16_1_fp32_row_avx2(const Q16_1Block *input,const float *gamma,
                                           float *output,std::size_t cols,float epsilon)
    {
        const auto bank=rmsnorm_operands::Q16ExpansionBorrow::bind(input,gamma,output,1,cols,epsilon);
        if(bank.empty())return;
        auto low=_mm256_setzero_pd(),high=_mm256_setzero_pd();
        for(std::size_t block=0;block<cols/32;++block)
        {
            const auto scale=_mm256_set1_ps(input[block].d);
            for(std::size_t offset=0;offset<32;offset+=8)
            {
                const auto codes=_mm_loadu_si128(reinterpret_cast<const __m128i*>(input[block].qs+offset));
                const auto value=x86_fp32::persistRounded(_mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(codes)),scale));
                const auto square=x86_fp32::persistRounded(_mm256_mul_ps(value,value));
                low=_mm256_add_pd(low,_mm256_cvtps_pd(_mm256_castps256_ps128(square)));
                high=_mm256_add_pd(high,_mm256_cvtps_pd(_mm256_extractf128_ps(square,1)));
            }
        }
        // Match the declared eight-lane native tree, never the worker partition.
        const auto folded=_mm_add_pd(
            _mm_add_pd(_mm256_castpd256_pd128(low),_mm256_extractf128_pd(low,1)),
            _mm_add_pd(_mm256_castpd256_pd128(high),_mm256_extractf128_pd(high,1)));
        const double sum=_mm_cvtsd_f64(folded)+_mm_cvtsd_f64(_mm_unpackhi_pd(folded,folded));
        if(!std::isfinite(sum))rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteSummary,
            "AVX2 Q16 RMS observed a nonfinite FP32-square summary");
        const float inverse=1.0f/std::sqrt(float(sum/double(cols))+epsilon);
        if(!std::isfinite(inverse))rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteInverse,
            "AVX2 Q16 RMS inverse is not representable");
        const auto inv=_mm256_set1_ps(inverse);
        for(std::size_t block=0;block<cols/32;++block)
        {
            const auto scale=_mm256_set1_ps(input[block].d);
            for(std::size_t offset=0;offset<32;offset+=8)
            {
                const auto column=block*32+offset;
                const auto codes=_mm_loadu_si128(reinterpret_cast<const __m128i*>(input[block].qs+offset));
                const auto value=x86_fp32::persistRounded(_mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(codes)),scale));
                const auto normalized=x86_fp32::persistRounded(_mm256_mul_ps(value,inv));
                _mm256_storeu_ps(output+column,_mm256_mul_ps(normalized,_mm256_loadu_ps(gamma+column)));
            }
        }
    }
#endif

    void rmsnorm_q16_1_fp32_row_scalar(
        const Q16_1Block *input,
        const float *gamma,
        float *output,
        std::size_t cols,
        float epsilon)
    {
        const auto bank=rmsnorm_operands::Q16ExpansionBorrow::bind(input,gamma,output,1,cols,epsilon);
        if(bank.empty())return;
        const size_t n_blocks = cols / 32;

        // This distinct program rounds each square to FP32 before widening;
        // do not replace it with floating RMSNorm's FP64-square reduction.
        // Pass 1: Compute sum of squares with double precision accumulation
        // Dequantize on-the-fly: val = scale * qs
        double sum_sq = 0.0;
        for (size_t blk = 0; blk < n_blocks; ++blk)
        {
            const Q16_1Block &block = input[blk];
            float scale = block.d;

            for (int i = 0; i < 32; ++i)
            {
                float val = scale * static_cast<float>(block.qs[i]);
                sum_sq += static_cast<double>(val * val);
            }
        }

        // Compute RMS and inverse through the original FP32-mean program.
        if(!std::isfinite(sum_sq))rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteSummary,
            "Scalar Q16 RMS observed a nonfinite FP32-square summary");
        float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
        float inv_rms = 1.0f / rms;
        if(!std::isfinite(inv_rms))rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteInverse,
            "Scalar Q16 RMS inverse is not representable");

        // Pass 2: Dequantize + normalize + gamma + write
        for (size_t blk = 0; blk < n_blocks; ++blk)
        {
            const Q16_1Block &block = input[blk];
            float scale = block.d;

            for (int i = 0; i < 32; ++i)
            {
                size_t idx = blk * 32 + i;
                float val = scale * static_cast<float>(block.qs[i]);
                output[idx] = val * inv_rms * gamma[idx];
            }
        }
    }

#if defined(__AVX512F__)
    void rmsnorm_q16_1_fp32_row_avx512(
        const Q16_1Block *input,
        const float *gamma,
        float *output,
        std::size_t cols,
        float epsilon)
    {
        const auto bank=rmsnorm_operands::Q16ExpansionBorrow::bind(input,gamma,output,1,cols,epsilon);
        if(bank.empty())return;
        const size_t n_blocks = cols / 32;

        // Pass 1: Vectorized sum of squares
        // Use double precision accumulation for numerical stability
        __m512d sum_sq_d_lo = _mm512_setzero_pd();
        __m512d sum_sq_d_hi = _mm512_setzero_pd();

        for (size_t blk = 0; blk < n_blocks; ++blk)
        {
            const Q16_1Block &block = input[blk];
            __m512 scale_vec = _mm512_set1_ps(block.d);

            // Load 16 int16 elements at a time → expand to 16 int32 → convert to 16 FP32
            // First half: qs[0..15]
            __m256i q_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block.qs));
            __m512i q32_lo = _mm512_cvtepi16_epi32(q_lo);
            __m512 f_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(q32_lo), scale_vec);

            // Second half: qs[16..31]
            __m256i q_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block.qs + 16));
            __m512i q32_hi = _mm512_cvtepi16_epi32(q_hi);
            __m512 f_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(q32_hi), scale_vec);

            // Compute squares (FP32)
            __m512 sq_lo = _mm512_mul_ps(f_lo, f_lo);
            __m512 sq_hi = _mm512_mul_ps(f_hi, f_hi);

            // Convert to double and accumulate (lower 8 floats → 8 doubles)
            __m256 sq_lo_lo = _mm512_castps512_ps256(sq_lo);
            __m256 sq_lo_hi = _mm512_extractf32x8_ps(sq_lo, 1);
            __m256 sq_hi_lo = _mm512_castps512_ps256(sq_hi);
            __m256 sq_hi_hi = _mm512_extractf32x8_ps(sq_hi, 1);

            sum_sq_d_lo = _mm512_add_pd(sum_sq_d_lo, _mm512_cvtps_pd(sq_lo_lo));
            sum_sq_d_hi = _mm512_add_pd(sum_sq_d_hi, _mm512_cvtps_pd(sq_lo_hi));
            sum_sq_d_lo = _mm512_add_pd(sum_sq_d_lo, _mm512_cvtps_pd(sq_hi_lo));
            sum_sq_d_hi = _mm512_add_pd(sum_sq_d_hi, _mm512_cvtps_pd(sq_hi_hi));
        }

        // Horizontal sum of doubles
        __m512d sum_total_d = _mm512_add_pd(sum_sq_d_lo, sum_sq_d_hi);
        double sum_sq = _mm512_reduce_add_pd(sum_total_d);

        // Compute RMS and inverse
        if(!std::isfinite(sum_sq))rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteSummary,
            "AVX512 Q16 RMS observed a nonfinite FP32-square summary");
        float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
        const float inverse=1.0f/rms;
        if(!std::isfinite(inverse))rms_numeric::rejectNumeric(rms_numeric::NumericFailure::NonfiniteInverse,
            "AVX512 Q16 RMS inverse is not representable");
        __m512 inv_rms_vec = _mm512_set1_ps(inverse);

        // Pass 2: Vectorized dequant + normalize + gamma + write
        for (size_t blk = 0; blk < n_blocks; ++blk)
        {
            const Q16_1Block &block = input[blk];
            __m512 scale_vec = _mm512_set1_ps(block.d);

            // Dequantize: qs → FP32
            __m256i q_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block.qs));
            __m256i q_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block.qs + 16));

            __m512 f_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q_lo)), scale_vec);
            __m512 f_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q_hi)), scale_vec);

            // Load gamma
            __m512 g_lo = _mm512_loadu_ps(gamma + blk * 32);
            __m512 g_hi = _mm512_loadu_ps(gamma + blk * 32 + 16);

            // Normalize and apply gamma: output = (val * inv_rms) * gamma
            __m512 out_lo = _mm512_mul_ps(_mm512_mul_ps(f_lo, inv_rms_vec), g_lo);
            __m512 out_hi = _mm512_mul_ps(_mm512_mul_ps(f_hi, inv_rms_vec), g_hi);

            // Store result
            _mm512_storeu_ps(output + blk * 32, out_lo);
            _mm512_storeu_ps(output + blk * 32 + 16, out_hi);
        }
    }
#else
    void rmsnorm_q16_1_fp32_row_avx512(
        const Q16_1Block *input,
        const float *gamma,
        float *output,
        std::size_t cols,
        float epsilon)
    {
        (void)input;(void)gamma;(void)output;(void)cols;(void)epsilon;
        throw std::runtime_error("Q16 RMS AVX512 entry requires an actual AVX512 compiler envelope");
    }
#endif

    void rmsnorm_q16_1_fp32_fused(
        const Q16_1Block *input,
        const float *gamma,
        float *output,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts)
    {
        const auto bank=rmsnorm_operands::Q16ExpansionBorrow::bind(input,gamma,output,rows,cols,epsilon);
        bank.requireMetadataReadOnly(&opts,sizeof(opts));
        if(opts.t5_compat_mode)
            rms_numeric::rejectPolicy("Q16 RMS FP32-square policy is distinct from FP32 T5 accumulation");
#if defined(__aarch64__) && defined(__ARM_NEON)
        if(use_native_rmsnorm(opts))
        {
            native_q16_matrix(bank,opts);
            return;
        }
#endif
        auto row_fn=rmsnorm_q16_1_fp32_row_scalar;
        if(!opts.force_scalar)
        {
            switch(activeISALevel())
            {
            case ISALevel::Scalar:break; // An explicitly selected scalar backend, not an ISA replacement.
#if defined(__AVX2__)
            case ISALevel::AVX2:row_fn=rmsnorm_q16_1_fp32_row_avx2;break;
#endif
#if defined(__AVX512F__)
            case ISALevel::AVX512:row_fn=rmsnorm_q16_1_fp32_row_avx512;break;
#endif
            default:rms_numeric::rejectRuntimePolicy("Q16 RMS active ISA has no matching native compiler implementation");
            }
        }
        const auto blocks=cols/32;
        auto work=[&](){
#pragma omp for schedule(static)
            for(std::ptrdiff_t row=0;row<std::ptrdiff_t(bank.empty()?0:rows);++row)
                row_fn(input+std::size_t(row)*blocks,gamma,output+std::size_t(row)*cols,cols,epsilon);
        };
        OMP_WORKSHARE_REGION_IF(work,omp_in_parallel()||want_parallel(rows,cols,opts));
    }

} // namespace llaminar2::primitives
