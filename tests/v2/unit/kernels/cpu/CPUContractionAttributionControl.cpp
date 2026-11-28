/**
 * @file CPUContractionAttributionControl.cpp
 * @brief Detection-power controls for the x86 packed-prefill FMA attribution witness.
 *
 * Compiled by tests/v2/scripts/cpu_contraction_proof.py with the actual command
 * of CPUNativeX86PackedPrefill.cpp per x86 envelope, once canonically and once
 * with -ffp-contract=off. A fused instruction is attributed to source only when
 * its whole function body is unchanged by -ffp-contract=off. attributionContracted*
 * writes a separate multiply and add intrinsic, which the canonical command
 * contracts: the witness must report it as compiler-introduced, otherwise a
 * clean prefill result is not evidence. attributionSource* writes an explicit
 * FMA intrinsic and must stay attributed to source.
 */
#include <immintrin.h>

#define ATTRIBUTION_CONTROL(SUFFIX, VECTOR, MUL, ADD, FMADD)                                                 \
    extern "C" __attribute__((noinline, used)) VECTOR attributionContracted##SUFFIX(VECTOR a, VECTOR b,       \
                                                                                     VECTOR c)                \
    {                                                                                                         \
        return ADD(MUL(a, b), c);                                                                             \
    }                                                                                                         \
    extern "C" __attribute__((noinline, used)) VECTOR attributionSource##SUFFIX(VECTOR a, VECTOR b, VECTOR c) \
    {                                                                                                         \
        return FMADD(a, b, c);                                                                                \
    }
ATTRIBUTION_CONTROL(AVX2, __m256, _mm256_mul_ps, _mm256_add_ps, _mm256_fmadd_ps)
#if defined(__AVX512F__)
ATTRIBUTION_CONTROL(AVX512, __m512, _mm512_mul_ps, _mm512_add_ps, _mm512_fmadd_ps)
#endif
