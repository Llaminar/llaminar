/**
 * @file ROCmQwenSparseOnlineRescale.cuh
 * @brief Exact online-max scale selection shared by production and native tests.
 *
 * Only known finite zero-difference exponentials are elided. Nonfinite cases
 * retain the original evaluator and the caller owns all original arithmetic.
 */
#pragma once
#include <hip/hip_runtime.h>
#include <cmath>
namespace llaminar2::qwen4exp::rocm_sparse_detail
{
/** @brief The two original online-max scales; no accumulated value is rewritten. */
struct SparseHalfOnlineScales
{
    float previous; ///< Scale applied by the original numerator/denominator FMAs.
    float tile; ///< Scale applied by the original tile multiplies.
};
/**
 * @brief Elide only exponentials whose original finite difference is signed zero.
 * @param maximum Original running maximum.
 * @param tileMaximum Original wave-reduced tile maximum.
 * @param denominator Original running denominator, retaining its exact zero guard.
 * @param nextMaximum The caller's unchanged original fmaxf result.
 * @return Original previous/tile scales with unchanged nonfinite expressions.
 * @details A finite winner contributes exp(+0/-0)==1 under the compiled native
 * evaluator, separately checked by runtime-input witnesses. A first finite tile
 * has running maximum−Inf/denominator0 and contributes (0,1). Every other
 * nonfinite case deliberately executes the literal original expressions. No
 * multiplication, FMA, reduction, store or publication is specialized here.
 */
__device__ __forceinline__ SparseHalfOnlineScales sparseHalfOnlineScales(
    float maximum,float tileMaximum,float denominator,float nextMaximum)
{
    if (__builtin_isfinite(maximum) && __builtin_isfinite(tileMaximum))
    {
        if (maximum < tileMaximum)
            return {denominator == 0.0f ? 0.0f : expf(maximum - nextMaximum), 1.0f};
        if (maximum > tileMaximum)
            return {denominator == 0.0f ? 0.0f : 1.0f, expf(tileMaximum - nextMaximum)};
        return {denominator == 0.0f ? 0.0f : 1.0f, 1.0f};
    }
    if (maximum == -INFINITY && denominator == 0.0f && __builtin_isfinite(tileMaximum))
        return {0.0f, 1.0f};
    return {denominator == 0.0f ? 0.0f : expf(maximum - nextMaximum),
            expf(tileMaximum - nextMaximum)};
}
}
