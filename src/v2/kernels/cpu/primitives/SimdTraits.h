/**
 * @file SimdTraits.h
 * @brief ISA abstraction layer for SIMD operations
 *
 * Provides template traits for different SIMD instruction sets (AVX512, AVX2, Scalar).
 * This enables ISA-agnostic GEMM kernel code through compile-time polymorphism.
 *
 * @author David Sanftenberg
 * @date October 2025
 */

#pragma once

#include <cstdint>
#include <cstring>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{
    namespace kernels
    {
        namespace simd
        {

            // ========== ISA TAG TYPES ==========

            /**
             * @brief Tag type for AVX512 ISA selection
             *
             * Requirements:
             * - __AVX512F__ must be defined
             * - CPU must support AVX-512 Foundation (Skylake-X, Ice Lake, Zen 4+)
             */
            struct AVX512Tag
            {
                static constexpr const char *name = "AVX512";
            };

            /**
             * @brief Tag type for AVX2 ISA selection
             *
             * Requirements:
             * - __AVX2__ must be defined
             * - CPU must support AVX2 + FMA3 (Haswell+, Zen 1+)
             */
            struct AVX2Tag
            {
                static constexpr const char *name = "AVX2";
            };

            /**
             * @brief Tag type for scalar fallback (portable, no SIMD)
             *
             * Used for:
             * - Testing on systems without SIMD support
             * - Reference implementation
             * - Validation of SIMD variants
             */
            struct ScalarTag
            {
                static constexpr const char *name = "Scalar";
            };

            /**
             * @brief Tag type for AVX512 VNNI ISA selection
             *
             * Requirements:
             * - __AVX512F__ and __AVX512VNNI__ must be defined
             * - CPU must support AVX512-VNNI (Cascade Lake+, Ice Lake+, Zen 4+)
             *
             * Provides INT8×INT8→INT32 GEMM via _mm512_dpbusd_epi32 instruction.
             * Used for quantized inference with 8-bit weights and activations.
             */
            struct AVX512VNNITag
            {
                static constexpr const char *name = "AVX512VNNI";
            };

            // ========== SIMD TRAITS BASE TEMPLATE ==========

            /**
             * @brief SIMD traits template - specialized for each ISA
             *
             * Provides:
             * - VectorType: SIMD register type (__m512, __m256, or mock struct)
             * - vector_width: Number of floats per vector (16, 8, or 1)
             * - isa_name: Human-readable ISA name
             * - SIMD operations: zero(), load(), fmadd(), reduce_add()
             *
             * @tparam ISA - ISA tag type (AVX512Tag, AVX2Tag, ScalarTag)
             */
            template <typename ISA>
            struct SimdTraits;

            // ========== AVX512 SPECIALIZATION ==========

#if defined(__AVX512F__)

            template <>
            struct SimdTraits<AVX512Tag>
            {
                using VectorType = __m512;
                static constexpr int vector_width = 16; // 16 floats per __m512
                static constexpr const char *isa_name = "AVX512";

                /**
                 * @brief Return zero vector
                 */
                static inline VectorType zero()
                {
                    return _mm512_setzero_ps();
                }

                /**
                 * @brief Load 16 floats from unaligned memory
                 */
                static inline VectorType load(const float *ptr)
                {
                    return _mm512_loadu_ps(ptr);
                }

                /**
                 * @brief Store 16 floats to unaligned memory
                 */
                static inline void store(float *ptr, VectorType v)
                {
                    _mm512_storeu_ps(ptr, v);
                }

                /**
                 * @brief Fused multiply-add: result = a * b + c
                 */
                static inline VectorType fmadd(VectorType a, VectorType b, VectorType c)
                {
                    return _mm512_fmadd_ps(a, b, c);
                }

                /**
                 * @brief Horizontal reduction: sum all 16 elements to scalar
                 */
                static inline float reduce_add(VectorType v)
                {
                    return _mm512_reduce_add_ps(v);
                }

                /**
                 * @brief Prefetch data into L1 cache
                 */
                static inline void prefetch_l1(const void *ptr)
                {
                    _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T0);
                }

                /**
                 * @brief Prefetch data into L2 cache
                 */
                static inline void prefetch_l2(const void *ptr)
                {
                    _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T1);
                }
            };

#endif // __AVX512F__

            // ========== AVX512 VNNI SPECIALIZATION (INT8×INT8→INT32) ==========

#if defined(__AVX512F__) && defined(__AVX512VNNI__)

            /**
             * @brief AVX512 VNNI traits for INT8×INT8→INT32 GEMM
             *
             * Key differences from FP32 AVX512:
             * - VectorType: __m512i (integer) not __m512 (float)
             * - Computes 4-way dot products via _mm512_dpbusd_epi32
             * - Accumulators are int32, not float
             * - No horizontal reduction (manual extraction needed)
             *
             * VNNI Instruction: _mm512_dpbusd_epi32(src, a, b)
             * Computes: src[i] += a[4i+0]*b[4i+0] + a[4i+1]*b[4i+1] +
             *                     a[4i+2]*b[4i+2] + a[4i+3]*b[4i+3]
             *
             * Each __m512i register holds:
             * - 64 int8 values (input) OR
             * - 16 int32 values (accumulator)
             */
            template <>
            struct SimdTraits<AVX512VNNITag>
            {
                using VectorType = __m512i;              // Integer SIMD register
                using AccumType = __m512i;               // int32 accumulator register
                static constexpr int vector_width = 64;  // 64 int8s per __m512i
                static constexpr int accum_width = 16;   // 16 int32s per __m512i
                static constexpr int dot_group_size = 4; // VNNI does 4-way dot products
                static constexpr const char *isa_name = "AVX512VNNI";

                /**
                 * @brief Return zero int32 accumulator vector
                 */
                static inline AccumType zero_i32()
                {
                    return _mm512_setzero_si512();
                }

                /**
                 * @brief Load 64 int8 values from unaligned memory
                 */
                static inline VectorType load_i8(const int8_t *ptr)
                {
                    return _mm512_loadu_si512(reinterpret_cast<const __m512i *>(ptr));
                }

                /**
                 * @brief Store 16 int32 values to unaligned memory
                 */
                static inline void store_i32(int32_t *ptr, AccumType v)
                {
                    _mm512_storeu_si512(reinterpret_cast<__m512i *>(ptr), v);
                }

                /**
                 * @brief VNNI dot product: src += a · b (4-way dot products)
                 *
                 * Computes 16 parallel 4-element dot products:
                 * src[i] += a[4i]*b[4i] + a[4i+1]*b[4i+1] + a[4i+2]*b[4i+2] + a[4i+3]*b[4i+3]
                 *
                 * @param src int32 accumulator (16 elements)
                 * @param a int8 vector (64 elements, treated as unsigned)
                 * @param b int8 vector (64 elements, treated as signed)
                 * @return Updated accumulator with dot products added
                 */
                static inline AccumType dpbusd(AccumType src, VectorType a, VectorType b)
                {
                    return _mm512_dpbusd_epi32(src, a, b);
                }

                /**
                 * @brief VNNI signed-signed dot product: src += a · b (4-way dot products, both signed)
                 *
                 * Uses instruction _mm512_dpbusds_epi32 which interprets BOTH a and b as signed int8.
                 * This is required for general INT8 activation/weight GEMM where values span [-128,127].
                 * The unsigned-signed variant (dpbusd) zero-extends the first operand and is only
                 * numerically correct when the first operand is guaranteed non-negative.
                 */
                static inline AccumType dpbusds(AccumType src, VectorType a, VectorType b)
                {
                    return _mm512_dpbusds_epi32(src, a, b);
                }

                /**
                 * @brief Extract int32 value at given index (0-15)
                 */
                static inline int32_t extract_i32(AccumType v, int index)
                {
                    alignas(64) int32_t tmp[16];
                    store_i32(tmp, v);
                    return tmp[index];
                }

                /**
                 * @brief Convert int32 accumulator to float with dequantization
                 *
                 * @param v int32 accumulator vector
                 * @param scale Scaling factor for dequantization
                 * @param zero_point Zero point for dequantization
                 * @return __m512 vector of 16 floats
                 */
                static inline __m512 cvt_i32_to_fp32_dequant(AccumType v, float scale, int32_t zero_point)
                {
                    // Convert int32 → float
                    __m512 fp32 = _mm512_cvtepi32_ps(v);

                    // Dequantize: (x - zero_point) * scale
                    __m512 zp_vec = _mm512_set1_ps(static_cast<float>(zero_point));
                    __m512 scale_vec = _mm512_set1_ps(scale);

                    fp32 = _mm512_sub_ps(fp32, zp_vec);
                    fp32 = _mm512_mul_ps(fp32, scale_vec);

                    return fp32;
                }

                /**
                 * @brief Prefetch data into L1 cache
                 */
                static inline void prefetch_l1(const void *ptr)
                {
                    _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T0);
                }

                /**
                 * @brief Prefetch data into L2 cache
                 */
                static inline void prefetch_l2(const void *ptr)
                {
                    _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T1);
                }
            };

#endif // __AVX512F__ && __AVX512VNNI__

            // ========== AVX2 SPECIALIZATION ==========

#if defined(__AVX2__)

            template <>
            struct SimdTraits<AVX2Tag>
            {
                using VectorType = __m256;
                static constexpr int vector_width = 8; // 8 floats per __m256
                static constexpr const char *isa_name = "AVX2";

                /**
                 * @brief Return zero vector
                 */
                static inline VectorType zero()
                {
                    return _mm256_setzero_ps();
                }

                /**
                 * @brief Load 8 floats from unaligned memory
                 */
                static inline VectorType load(const float *ptr)
                {
                    return _mm256_loadu_ps(ptr);
                }

                /**
                 * @brief Store 8 floats to unaligned memory
                 */
                static inline void store(float *ptr, VectorType v)
                {
                    _mm256_storeu_ps(ptr, v);
                }

                /**
                 * @brief Fused multiply-add: result = a * b + c
                 *
                 * Note: All AVX2 CPUs have FMA3 (Haswell+), so this is always available
                 */
                static inline VectorType fmadd(VectorType a, VectorType b, VectorType c)
                {
                    return _mm256_fmadd_ps(a, b, c);
                }

                /**
                 * @brief Horizontal reduction: sum all 8 elements to scalar
                 *
                 * Implementation:
                 * 1. Extract 128-bit halves
                 * 2. Add halves vertically
                 * 3. Horizontal add twice (4→2→1)
                 */
                static inline float reduce_add(VectorType v)
                {
                    // Extract low and high 128-bit lanes
                    __m128 lo = _mm256_castps256_ps128(v);
                    __m128 hi = _mm256_extractf128_ps(v, 1);

                    // Add lanes: [a0..a3] + [a4..a7]
                    __m128 sum = _mm_add_ps(lo, hi);

                    // Horizontal add: [a0+a1, a2+a3, a0+a1, a2+a3]
                    sum = _mm_hadd_ps(sum, sum);

                    // Horizontal add: [a0+a1+a2+a3, *, a0+a1+a2+a3, *]
                    sum = _mm_hadd_ps(sum, sum);

                    // Extract scalar
                    return _mm_cvtss_f32(sum);
                }

                /**
                 * @brief Prefetch data into L1 cache
                 */
                static inline void prefetch_l1(const void *ptr)
                {
                    _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T0);
                }

                /**
                 * @brief Prefetch data into L2 cache
                 */
                static inline void prefetch_l2(const void *ptr)
                {
                    _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T1);
                }
            };

#endif // __AVX2__

            // ========== SCALAR FALLBACK SPECIALIZATION ==========

            /**
             * @brief Mock SIMD vector for scalar fallback
             *
             * Uses array of 16 floats to mimic SIMD register behavior.
             * Useful for testing and validation on non-SIMD systems.
             */
            struct ScalarVector
            {
                alignas(64) float data[16];

                ScalarVector()
                {
                    std::memset(data, 0, sizeof(data));
                }

                explicit ScalarVector(float val)
                {
                    for (int i = 0; i < 16; ++i)
                    {
                        data[i] = val;
                    }
                }
            };

            template <>
            struct SimdTraits<ScalarTag>
            {
                using VectorType = ScalarVector;
                static constexpr int vector_width = 1; // Process 1 element at a time
                static constexpr const char *isa_name = "Scalar";

                /**
                 * @brief Return zero vector
                 */
                static inline VectorType zero()
                {
                    return ScalarVector(0.0f);
                }

                /**
                 * @brief Load from memory (scalar: just reads first element)
                 */
                static inline VectorType load(const float *ptr)
                {
                    ScalarVector v;
                    v.data[0] = ptr[0];
                    return v;
                }

                /**
                 * @brief Store to memory (scalar: just writes first element)
                 */
                static inline void store(float *ptr, VectorType v)
                {
                    ptr[0] = v.data[0];
                }

                /**
                 * @brief Fused multiply-add: result = a * b + c
                 */
                static inline VectorType fmadd(VectorType a, VectorType b, VectorType c)
                {
                    ScalarVector result;
                    result.data[0] = a.data[0] * b.data[0] + c.data[0];
                    return result;
                }

                /**
                 * @brief Horizontal reduction: return scalar value
                 */
                static inline float reduce_add(VectorType v)
                {
                    return v.data[0];
                }

                /**
                 * @brief Prefetch (no-op for scalar)
                 */
                static inline void prefetch_l1(const void *ptr)
                {
                    (void)ptr; // Suppress unused warning
                }

                /**
                 * @brief Prefetch (no-op for scalar)
                 */
                static inline void prefetch_l2(const void *ptr)
                {
                    (void)ptr; // Suppress unused warning
                }
            };

            // ========== COMPILE-TIME ISA AVAILABILITY CHECKS ==========

            /**
             * @brief Check if ISA is available at compile time
             */
            template <typename ISA>
            struct IsaAvailable
            {
                static constexpr bool value = false;
            };

#if defined(__AVX512F__)
            template <>
            struct IsaAvailable<AVX512Tag>
            {
                static constexpr bool value = true;
            };
#endif

#if defined(__AVX2__)
            template <>
            struct IsaAvailable<AVX2Tag>
            {
                static constexpr bool value = true;
            };
#endif

            template <>
            struct IsaAvailable<ScalarTag>
            {
                static constexpr bool value = true; // Always available
            };

        } // namespace simd
    } // namespace kernels
} // namespace llaminar2
