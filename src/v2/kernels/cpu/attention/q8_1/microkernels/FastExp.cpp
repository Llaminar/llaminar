/**
 * @file FastExp.cpp
 * @brief Fast exponential implementations for softmax
 * @author David Sanftenberg
 */

#include "FastExp.h"
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar::v2::kernels::microkernels {

float fast_exp_range_reduced(float x) {
    // Use identity: exp(x) = 2^(x * log2(e))
    // Split into integer part (exact) and fractional part (polynomial)
    
    constexpr float LOG2E = 1.4426950408889634f;  // log2(e)
    constexpr float LN2 = 0.6931471805599453f;    // ln(2)
    
    // Clamp to avoid overflow/underflow
    x = clamp_for_exp(x);
    
    // Convert to base 2: x * log2(e)
    float y = x * LOG2E;
    
    // Split into integer and fractional parts
    float yi = std::floor(y);
    float yf = y - yi;
    
    // Compute 2^yf using polynomial (yf in [0, 1))
    // 2^x ≈ 1 + x*ln(2) + x²*ln(2)²/2 + x³*ln(2)³/6 + x⁴*ln(2)⁴/24
    // Or use direct polynomial fit for 2^x
    constexpr float c0 = 1.0f;
    constexpr float c1 = 0.6931471806f;   // ln(2)
    constexpr float c2 = 0.2402265069f;   // ln(2)²/2
    constexpr float c3 = 0.0555041086f;   // ln(2)³/6
    constexpr float c4 = 0.0096181291f;   // ln(2)⁴/24
    constexpr float c5 = 0.0013333558f;   // ln(2)⁵/120
    
    float exp_frac = c0 + yf * (c1 + yf * (c2 + yf * (c3 + yf * (c4 + yf * c5))));
    
    // Compute 2^yi using bit manipulation
    // 2^n = (127 + n) << 23 for IEEE 754 float (as exponent bits)
    int32_t n = static_cast<int32_t>(yi);
    
    // Clamp n to valid exponent range
    if (n < -126) return 0.0f;
    if (n > 127) return std::numeric_limits<float>::infinity();
    
    // Construct 2^n
    uint32_t exp_int_bits = static_cast<uint32_t>(127 + n) << 23;
    float exp_int;
    std::memcpy(&exp_int, &exp_int_bits, sizeof(float));
    
    return exp_int * exp_frac;
}

#if defined(__AVX512F__)

__m512 fast_exp_poly_avx512(__m512 x) {
    // Taylor series coefficients
    const __m512 c0 = _mm512_set1_ps(1.0f);
    const __m512 c1 = _mm512_set1_ps(1.0f);
    const __m512 c2 = _mm512_set1_ps(0.5f);
    const __m512 c3 = _mm512_set1_ps(0.16666667f);
    const __m512 c4 = _mm512_set1_ps(0.041666668f);
    const __m512 c5 = _mm512_set1_ps(0.0083333338f);
    
    // Horner's method
    __m512 result = c5;
    result = _mm512_fmadd_ps(result, x, c4);
    result = _mm512_fmadd_ps(result, x, c3);
    result = _mm512_fmadd_ps(result, x, c2);
    result = _mm512_fmadd_ps(result, x, c1);
    result = _mm512_fmadd_ps(result, x, c0);
    
    return result;
}

__m512 fast_exp_avx512(__m512 x) {
    // Range reduction: exp(x) = 2^(x * log2(e)) = 2^n * 2^f
    // where n = floor(x * log2(e)) and f = fractional part
    
    const __m512 LOG2E = _mm512_set1_ps(1.4426950408889634f);
    const __m512 LN2 = _mm512_set1_ps(0.6931471805599453f);
    
    // Clamp input to avoid overflow
    const __m512 min_val = _mm512_set1_ps(-88.0f);
    const __m512 max_val = _mm512_set1_ps(88.0f);
    x = _mm512_max_ps(x, min_val);
    x = _mm512_min_ps(x, max_val);
    
    // y = x * log2(e)
    __m512 y = _mm512_mul_ps(x, LOG2E);
    
    // n = floor(y)
    __m512 n = _mm512_floor_ps(y);
    
    // f = y - n (fractional part in [0, 1))
    __m512 f = _mm512_sub_ps(y, n);
    
    // Compute 2^f using polynomial
    // Coefficients for 2^x approximation on [0, 1]
    const __m512 c0 = _mm512_set1_ps(1.0f);
    const __m512 c1 = _mm512_set1_ps(0.6931471806f);
    const __m512 c2 = _mm512_set1_ps(0.2402265069f);
    const __m512 c3 = _mm512_set1_ps(0.0555041086f);
    const __m512 c4 = _mm512_set1_ps(0.0096181291f);
    const __m512 c5 = _mm512_set1_ps(0.0013333558f);
    
    __m512 exp_f = c5;
    exp_f = _mm512_fmadd_ps(exp_f, f, c4);
    exp_f = _mm512_fmadd_ps(exp_f, f, c3);
    exp_f = _mm512_fmadd_ps(exp_f, f, c2);
    exp_f = _mm512_fmadd_ps(exp_f, f, c1);
    exp_f = _mm512_fmadd_ps(exp_f, f, c0);
    
    // Compute 2^n by adding n to exponent
    // 2^n = (127 + n) << 23 in IEEE 754
    __m512i n_int = _mm512_cvtps_epi32(n);
    __m512i bias = _mm512_set1_epi32(127);
    __m512i exp_bits = _mm512_add_epi32(n_int, bias);
    exp_bits = _mm512_slli_epi32(exp_bits, 23);
    __m512 exp_n = _mm512_castsi512_ps(exp_bits);
    
    // Result = 2^n * 2^f
    return _mm512_mul_ps(exp_n, exp_f);
}

#endif // __AVX512F__

} // namespace llaminar::v2::kernels::microkernels
