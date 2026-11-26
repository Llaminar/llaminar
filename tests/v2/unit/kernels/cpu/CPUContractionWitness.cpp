/**
 * @file CPUContractionWitness.cpp
 * @brief Isolated disassembly witnesses of actual x86 contribution helpers.
 *
 * Compiled by tests/v2/scripts/cpu_contraction_proof.py with the production
 * core's actual command per x86 envelope, canonically and with an explicit
 * -ffp-contract=fast setting. Each externally visible function owns one policy, so an FMA in
 * an expert symbol cannot be confused with an intentional backend-native FMA.
 */
#include "kernels/cpu/gemm/CPUNativeQuantisedContributionContract.h"
using namespace llaminar2;
using namespace llaminar2::cpu::native_quantised;
#define WITNESS(SUFFIX, VECTOR, INTEGER, CONVERT) \
extern "C" __attribute__((noinline, used)) VECTOR expert##SUFFIX( \
    VECTOR a, INTEGER d, VECTOR s, float x) { \
    return accumulateSingleScaleBlock##SUFFIX(a, d, s, x, \
        CPUProjectionNumericalPolicy::GPUAlignedExpert); } \
extern "C" __attribute__((noinline, used)) VECTOR native##SUFFIX( \
    VECTOR a, INTEGER d, VECTOR s, float x) { \
    return accumulateSingleScaleBlock##SUFFIX(a, d, s, x, \
        CPUProjectionNumericalPolicy::BackendNative); } \
extern "C" __attribute__((noinline, used)) VECTOR negative##SUFFIX( \
    VECTOR a, INTEGER d, VECTOR s, float x) { \
    return a + (s * x) * CONVERT(d); }
WITNESS(AVX2, __m256, __m256i, _mm256_cvtepi32_ps)
#if defined(__AVX512F__)
WITNESS(AVX512, __m512, __m512i, _mm512_cvtepi32_ps)
#endif
