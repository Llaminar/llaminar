#pragma once
#include <cstddef>
#include <cstdint>
#include "BlockStructures.h"

namespace llaminar2::detail
{

    float q8_0_find_max_abs_scalar(const Q8_0Block *row_blocks, size_t blocks_per_row);

#if defined(__AVX2__)
    float q8_0_find_max_abs_avx2(const Q8_0Block *row_blocks, size_t blocks_per_row);
#endif

    void q8_0_requantize_scalar(const Q8_0Block *row_blocks, size_t blocks_per_row,
                                float inv_row_scale, int8_t *output);

#if defined(__AVX2__)
    void q8_0_requantize_avx2(const Q8_0Block *row_blocks, size_t blocks_per_row,
                              float inv_row_scale, int8_t *output);
#endif

#if defined(__AVX512F__)
    void q8_0_requantize_avx512(const Q8_0Block *row_blocks, size_t blocks_per_row,
                                float inv_row_scale, int8_t *output);
#endif

} // namespace llaminar2::detail
