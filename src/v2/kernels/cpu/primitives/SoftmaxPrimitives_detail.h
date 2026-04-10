#pragma once
#include <cstddef>

namespace llaminar2::primitives::detail
{

    void softmax_row_scalar(float *row, int cols, long long r,
                            bool causal, float scale, bool use_fast_exp);

#if defined(__AVX2__) && !defined(__AVX512F__)
    void softmax_row_avx2(float *row, int cols, long long r,
                          bool causal, float scale, bool use_fast_exp);
#endif

#if defined(__AVX512F__)
    void softmax_row_avx512(float *row, int cols, long long r,
                            bool causal, float scale, bool use_fast_exp);
#endif

} // namespace llaminar2::primitives::detail
