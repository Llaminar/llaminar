/**
 * @file FastExp.h
 * @brief Microkernel μK5: Fast exponential approximation for softmax
 * @author David Sanftenberg
 * 
 * Provides fast exp() approximation suitable for softmax computation.
 * Uses polynomial approximation (Taylor series or minimax) for speed.
 * 
 * Accuracy target: < 1% relative error in softmax-relevant range [-10, 0]
 */

#pragma once

#include <cmath>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar::v2::kernels::microkernels {

/**
 * @brief Reference fast exp using std::exp
 * 
 * Baseline for accuracy comparison.
 * 
 * @param x Input value
 * @return exp(x)
 */
inline float fast_exp_ref(float x) {
    return std::exp(x);
}

/**
 * @brief Polynomial approximation of exp(x)
 * 
 * Uses 5th-order Taylor series:
 *   exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120
 * 
 * Valid for small |x|. For large negative x (softmax with large differences),
 * the result approaches 0 which is acceptable.
 * 
 * @param x Input value (best accuracy for |x| < 2)
 * @return Approximate exp(x)
 */
inline float fast_exp_poly(float x) {
    // Coefficients: 1/n! for n = 0..5
    constexpr float c0 = 1.0f;
    constexpr float c1 = 1.0f;
    constexpr float c2 = 0.5f;           // 1/2
    constexpr float c3 = 0.16666667f;    // 1/6
    constexpr float c4 = 0.041666668f;   // 1/24
    constexpr float c5 = 0.0083333338f;  // 1/120
    
    // Horner's method for efficiency
    float result = c5;
    result = result * x + c4;
    result = result * x + c3;
    result = result * x + c2;
    result = result * x + c1;
    result = result * x + c0;
    
    return result;
}

/**
 * @brief Range-reduced exp using exp2
 * 
 * Uses identity: exp(x) = 2^(x * log2(e))
 * Then splits into integer and fractional parts for accuracy.
 * 
 * @param x Input value
 * @return exp(x)
 */
float fast_exp_range_reduced(float x);

#if defined(__AVX512F__)

/**
 * @brief AVX-512 vectorized polynomial exp (16 floats)
 * 
 * Computes exp() for 16 floats simultaneously using Taylor polynomial.
 * 
 * @param x Input vector of 16 floats
 * @return Vector of exp(x[i]) for i in [0,16)
 */
__m512 fast_exp_poly_avx512(__m512 x);

/**
 * @brief AVX-512 vectorized range-reduced exp (16 floats)
 * 
 * More accurate than polynomial for large |x|.
 * 
 * @param x Input vector
 * @return Vector of exp(x[i])
 */
__m512 fast_exp_avx512(__m512 x);

#endif // __AVX512F__

/**
 * @brief Dispatch to best available scalar implementation
 */
inline float fast_exp(float x) {
    // For now, use range-reduced for accuracy
    // Can switch to poly for speed if accuracy is acceptable
    return fast_exp_range_reduced(x);
}

/**
 * @brief Clamp input to safe range before exp
 * 
 * Prevents overflow/underflow. For softmax, very negative values
 * just become 0 anyway.
 * 
 * @param x Input value
 * @return Clamped value safe for exp()
 */
inline float clamp_for_exp(float x) {
    constexpr float MIN_EXP_ARG = -88.0f;  // exp(-88) ≈ 0
    constexpr float MAX_EXP_ARG = 88.0f;   // exp(88) ≈ 1.6e38
    return (x < MIN_EXP_ARG) ? MIN_EXP_ARG : ((x > MAX_EXP_ARG) ? MAX_EXP_ARG : x);
}

} // namespace llaminar::v2::kernels::microkernels
