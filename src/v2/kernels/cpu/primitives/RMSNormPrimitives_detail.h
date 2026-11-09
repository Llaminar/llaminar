/**
 * @file RMSNormPrimitives_detail.h
 * @brief Diagnostic declarations for existing scalar and compiled native RMS leaves.
 *
 * This header exposes actual arithmetic programs to focused precision tests,
 * never absent-ISA substitutes. Original INT32 row normalization/publication
 * goes through the checked public publisher; the obsolete decoded-FP32-row
 * helpers are deliberately not another production or diagnostic execution path.
 */
#pragma once
#include <cstddef>
#include <cstdint>

namespace llaminar2::primitives::detail
{

    // ===== FP32 sum-of-squares (NS1) =====

    /** @brief Observe increasing-column FP64 FMA over original FP32 words.
     * @param row Complete original row; @param cols Logical width; @return Explicit scalar squared summary.
     * @details Finite FP32 squares are exact in FP64; the scalar summation
     * order is fixed, distinct from each native ISA's fixed reduction tree.
     */
    double compute_sumsq_scalar(const float *row, std::size_t cols);

#if defined(__AVX2__) && !defined(__AVX512F__)
    /** @brief Observe the compiled AVX2 FP32 summary.
     * @param row Original row; @param cols Logical width; @return Native fixed-tree summary.
     */
    double compute_sumsq_avx2(const float *row, std::size_t cols);
#endif

#if defined(__AVX512F__)
    /** @brief Observe the compiled AVX512 FP32 summary.
     * @param row Original row; @param cols Logical width; @return Declared native summary.
     */
    double compute_sumsq_avx512(const float *row, std::size_t cols);
#endif

    // ===== INT32 RMS sum-of-squares (NS2) =====

    /** @brief Observe increasing-column FP64 FMA over original full-width integers.
     * @param row Original integers; @param cols Logical width; @return Explicit scalar summary.
     */
    double compute_rms_sq_int32_scalar(const int32_t *row, std::size_t cols);

#if defined(__AVX2__)
    /** @brief Observe eight native FP64 chains without an intermediate FP32 conversion.
     * @param row Original integers; @param cols Logical width; @return Fixed eight-chain summary.
     */
    double compute_rms_sq_int32_avx2(const int32_t *row, std::size_t cols);
#endif

#if defined(__AVX512F__)
    /** @brief Observe the existing distinct AVX512 FP64 integer program.
     * @param row Original integers; @param cols Logical width; @return Original native summary.
     */
    double compute_rms_sq_int32_avx512(const int32_t *row, std::size_t cols);
#endif

    // ===== BF16/FP16 RMSNorm row kernels (NS4) =====

    /** @brief Execute the scalar BF16 diagnostic row program.
     * @param src_row Original BF16 words; @param gamma Original weights; @param dst_row Result words.
     * @param cols Logical width; @param epsilon Authored inverse policy.
     */
    void bf16_rmsnorm_row_scalar(const uint16_t *src_row, const float *gamma,
                                 uint16_t *dst_row, std::size_t cols, float epsilon);

#if defined(__AVX512F__)
    /** @brief Execute the compiled AVX512 BF16 row program.
     * @param src_row Original words; @param gamma Original weights; @param dst_row Result words.
     * @param cols Logical width; @param epsilon Original inverse policy.
     */
    void bf16_rmsnorm_row_avx512(const uint16_t *src_row, const float *gamma,
                                 uint16_t *dst_row, std::size_t cols, float epsilon);
#endif

    /** @brief Execute the scalar FP16 diagnostic row program.
     * @param src_row Original half words; @param gamma Original weights; @param dst_row Result words.
     * @param cols Logical width; @param epsilon Original inverse policy.
     */
    void fp16_rmsnorm_row_scalar(const uint16_t *src_row, const float *gamma,
                                 uint16_t *dst_row, std::size_t cols, float epsilon);

#if defined(__AVX512F__)
    /** @brief Execute the compiled AVX512 FP16 row program.
     * @param src_row Original half words; @param gamma Original weights; @param dst_row Result words.
     * @param cols Logical width; @param epsilon Original inverse policy.
     */
    void fp16_rmsnorm_row_avx512(const uint16_t *src_row, const float *gamma,
                                 uint16_t *dst_row, std::size_t cols, float epsilon);
#endif

#if defined(__AVX2__) && defined(__F16C__)
    /** @brief Execute the compiled AVX2/F16C row program, not an ARM codec oracle.
     * @param src_row Original half words; @param gamma Original weights; @param dst_row Result words.
     * @param cols Logical width; @param epsilon Original inverse policy.
     */
    void fp16_rmsnorm_row_avx2(const uint16_t *src_row, const float *gamma,
                               uint16_t *dst_row, std::size_t cols, float epsilon);
#endif

    // ===== Q8_1 integer-space RMSNorm helpers (NS5) =====

    /** @brief Observe the scalar signed-code energy.
     * @param qs Complete original32-code block; @return Exact bounded integer energy.
     */
    int32_t compute_int8_sumsq_scalar(const int8_t *qs);

#if defined(__AVX2__)
    /** @brief Observe the compiled AVX2 signed-code energy.
     * @param qs Complete original32-code block; @return Exact bounded integer energy.
     */
    int32_t compute_int8_sumsq_avx2(const int8_t *qs);
#endif

#if defined(__AVX512F__)
    /** @brief Observe the compiled AVX512 signed-code energy.
     * @param qs Complete original32-code block; @return Exact bounded integer energy.
     */
    int32_t compute_int8_sumsq_avx512(const int8_t *qs);
#endif

} // namespace llaminar2::primitives::detail
