#pragma once
#include <cstddef>
#include <cstdint>

namespace llaminar2::primitives::detail
{

    // ===== FP32 sum-of-squares (NS1) =====

    double compute_sumsq_scalar(const float *row, std::size_t cols);

#if defined(__AVX2__) && !defined(__AVX512F__)
    double compute_sumsq_avx2(const float *row, std::size_t cols);
#endif

#if defined(__AVX512F__)
    double compute_sumsq_avx512(const float *row, std::size_t cols);
#endif

    // ===== INT32 RMS sum-of-squares (NS2) =====

    double compute_rms_sq_int32_scalar(const int32_t *row, std::size_t cols);

#if defined(__AVX2__)
    double compute_rms_sq_int32_avx2(const int32_t *row, std::size_t cols);
#endif

#if defined(__AVX512F__)
    double compute_rms_sq_int32_avx512(const int32_t *row, std::size_t cols);
#endif

    // ===== INT32→FP32 normalization (NS3) =====

    void int32_to_fp32_normalize_scalar(const int32_t *src, float *dst,
                                        float rms_inv, const float *gamma,
                                        std::size_t cols);

#if defined(__AVX2__) && !defined(__AVX512F__)
    void int32_to_fp32_normalize_avx2(const int32_t *src, float *dst,
                                      float rms_inv, const float *gamma,
                                      std::size_t cols);
#endif

#if defined(__AVX512F__)
    void int32_to_fp32_normalize_avx512(const int32_t *src, float *dst,
                                        float rms_inv, const float *gamma,
                                        std::size_t cols);
#endif

    // ===== BF16/FP16 RMSNorm row kernels (NS4) =====

    void bf16_rmsnorm_row_scalar(const uint16_t *src_row, const float *gamma,
                                 uint16_t *dst_row, std::size_t cols, float epsilon);

#if defined(__AVX512F__)
    void bf16_rmsnorm_row_avx512(const uint16_t *src_row, const float *gamma,
                                 uint16_t *dst_row, std::size_t cols, float epsilon);
#endif

    void fp16_rmsnorm_row_scalar(const uint16_t *src_row, const float *gamma,
                                 uint16_t *dst_row, std::size_t cols, float epsilon);

#if defined(__AVX512F__)
    void fp16_rmsnorm_row_avx512(const uint16_t *src_row, const float *gamma,
                                 uint16_t *dst_row, std::size_t cols, float epsilon);
#endif

#if defined(__AVX2__) && defined(__F16C__)
    void fp16_rmsnorm_row_avx2(const uint16_t *src_row, const float *gamma,
                               uint16_t *dst_row, std::size_t cols, float epsilon);
#endif

    // ===== Q8_1 integer-space RMSNorm helpers (NS5) =====

    int32_t compute_int8_sumsq_scalar(const int8_t *qs);

#if defined(__AVX2__)
    int32_t compute_int8_sumsq_avx2(const int8_t *qs);
#endif

#if defined(__AVX512F__)
    int32_t compute_int8_sumsq_avx512(const int8_t *qs);
#endif

} // namespace llaminar2::primitives::detail
