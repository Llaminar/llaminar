#pragma once
#include <cstdint>
#include "../../../tensors/BlockStructures.h"

namespace llaminar2::primitives::detail
{

    // ===== Partial RoPE head rotation (NS2) =====

    void rope_rotate_head_scalar(float *head_ptr, const float *cos_ptr,
                                 const float *sin_ptr, int half_rotary);

#if defined(__AVX2__)
    void rope_rotate_head_avx2(float *head_ptr, const float *cos_ptr,
                               const float *sin_ptr, int half_rotary);
#endif

#if defined(__AVX512F__)
    void rope_rotate_head_avx512(float *head_ptr, const float *cos_ptr,
                                 const float *sin_ptr, int half_rotary);
#endif

    // ===== Q8_1→Q16_1 block pair rotation (NS4) =====

    void rotate_q8_1_block_pair_to_q16_1_scalar(
        const Q8_1Block &blockA, const Q8_1Block &blockB,
        Q16_1Block &outA, Q16_1Block &outB,
        const int16_t *cos_q15, const int16_t *sin_q15,
        float common_scale);

#if defined(__AVX2__)
    void rotate_q8_1_block_pair_to_q16_1_avx2(
        const Q8_1Block &blockA, const Q8_1Block &blockB,
        Q16_1Block &outA, Q16_1Block &outB,
        const int16_t *cos_q15, const int16_t *sin_q15,
        float common_scale);
#endif

#if defined(__AVX512F__)
    void rotate_q8_1_block_pair_to_q16_1_avx512(
        const Q8_1Block &blockA, const Q8_1Block &blockB,
        Q16_1Block &outA, Q16_1Block &outB,
        const int16_t *cos_q15, const int16_t *sin_q15,
        float common_scale);
#endif

} // namespace llaminar2::primitives::detail
